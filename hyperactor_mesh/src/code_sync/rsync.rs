/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::io::ErrorKind;
use std::net::SocketAddr;
use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use async_trait::async_trait;
use futures::try_join;
use hyperactor::Actor;
use hyperactor::Handler;
use hyperactor::Named;
use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;
use nix::sys::signal;
use nix::sys::signal::Signal;
use nix::unistd::Pid;
use serde::Deserialize;
use serde::Serialize;
use tempfile::TempDir;
use tokio::fs;
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tokio::process::Child;
use tokio::process::Command;

use crate::actor_mesh::ActorMesh;
use crate::code_sync::WorkspaceLocation;
use crate::connect::Connect;
use crate::connect::accept;
use crate::connect::connect_mesh;

pub async fn do_rsync(addr: &SocketAddr, workspace: &Path) -> Result<()> {
    let output = Command::new("rsync")
        .arg("--quiet")
        .arg("--archive")
        .arg("--delete")
        // By setting these flags, we make `rsync` immune to multiple invocations
        // targeting the same dir, which can happen if we don't take care to only
        // allow one worker on a given host to do the `rsync`.
        .arg("--delete-after")
        .arg("--delay-updates")
        .arg("--exclude=.rsync-tmp.*")
        .arg(format!("--partial-dir=.rsync-tmp.{}", addr.port()))
        .arg(format!("{}/", workspace.display()))
        .arg(format!("rsync://{}/workspace", addr))
        .stderr(Stdio::piped())
        .output()
        .await?;
    output
        .status
        .exit_ok()
        .with_context(|| format!("rsync failed: {}", String::from_utf8_lossy(&output.stderr)))?;
    Ok(())
}

#[derive(Debug)]
pub struct RsyncDaemon {
    child: Child,
    #[allow(unused)]
    state: TempDir,
    addr: SocketAddr,
}

impl RsyncDaemon {
    pub async fn spawn(listener: TcpListener, workspace: &Path) -> Result<Self> {
        let state = TempDir::with_prefix("rsyncd.")?;

        // Write rsync config file
        // TODO(agallagher): We can setup a secrets file to provide some measure of
        // security and prevent stray rsync calls from hitting the server.
        let content = format!(
            r#"\
[workspace]
    path = {workspace}
    use chroot = no
    list = no
    read only = false
    write only = true
    uid = {uid}
    hosts allow = localhost
"#,
            workspace = workspace.display(),
            uid = nix::unistd::getuid().as_raw(),
        );
        let config = state.path().join("rsync.config");
        fs::write(&config, content).await?;

        // Find free port.  This is potentially racy, as some process could
        // potentially bind to this port in between now and when `rsync` starts up
        // below.  But I'm not sure a better way to do this, as rsync doesn't appear
        // to support `rsync --sockopts=SO_PORTREUSE` (to share this port we've
        // reserved) or `--port=0` (to pick a free port -- it'll just always use
        // 873).
        let addr = listener.local_addr()?;
        std::mem::drop(listener);

        // Spawn the rsync daemon.
        let mut child = Command::new("rsync")
            .arg("--daemon")
            .arg("--no-detach")
            .arg(format!("--address={}", addr.ip()))
            .arg(format!("--port={}", addr.port()))
            .arg(format!("--config={}", config.display()))
            //.arg(format!("--log-file={}/log", state.path().display()))
            .arg("--log-file=/dev/stderr")
            .kill_on_drop(true)
            .spawn()?;

        // Wait until the rsync daemon is ready to connect via polling it (I tried polling
        // the log file to wait for the "listening" log line, but that gets prevented *before*
        // it actually starts the listening loop).
        tokio::select! {
            res = child.wait() => bail!("unexpected early exit: {:?}", res),
            res = async {
                loop {
                    match TcpStream::connect(addr).await {
                        Err(err) if err.kind() == ErrorKind::ConnectionRefused => {
                            RealClock.sleep(Duration::from_millis(1)).await
                        }
                        Err(err) => return Err(err.into()),
                        Ok(_) => break,
                    }
                }
                anyhow::Ok(())
            } => res?,
        }

        Ok(Self { child, state, addr })
    }

    pub fn addr(&self) -> &SocketAddr {
        &self.addr
    }

    pub async fn shutdown(mut self) -> Result<()> {
        let id = self.child.id().context("missing pid")?;
        let pid = Pid::from_raw(id as i32);
        signal::kill(pid, Signal::SIGINT)?;
        let status = self.child.wait().await?;
        // rsync exists with 20 when sent SIGINT.
        ensure!(status.code() == Some(20));
        Ok(())
    }
}

#[derive(Debug, Named, Serialize, Deserialize)]
pub struct RsyncParams {
    pub workspace: WorkspaceLocation,
}

#[derive(Debug)]
#[hyperactor::export(
    spawn = true,
    handlers = [Connect { cast = true }],
)]
pub struct RsyncActor {
    daemon: RsyncDaemon,
}

#[async_trait]
impl Actor for RsyncActor {
    type Params = RsyncParams;

    async fn new(RsyncParams { workspace }: Self::Params) -> Result<Self> {
        let workspace = workspace.resolve()?;
        let daemon = RsyncDaemon::spawn(TcpListener::bind(("::1", 0)).await?, &workspace).await?;
        Ok(Self { daemon })
    }
}

#[async_trait]
impl Handler<Connect> for RsyncActor {
    async fn handle(
        &mut self,
        this: &hyperactor::Context<Self>,
        message: Connect,
    ) -> Result<(), anyhow::Error> {
        let (mut local, mut stream) = try_join!(
            async { Ok(TcpStream::connect(self.daemon.addr()).await?) },
            async {
                let (rd, wr) = accept(this, message).await?;
                anyhow::Ok(tokio::io::join(rd, wr))
            }
        )?;
        tokio::io::copy_bidirectional(&mut local, &mut stream).await?;
        Ok(())
    }
}

pub async fn rsync_mesh<M>(actor_mesh: M, workspace: PathBuf) -> Result<()>
where
    M: ActorMesh<Actor = RsyncActor>,
{
    connect_mesh(actor_mesh, async move |rd, wr| {
        let workspace = workspace.clone();
        let listener = TcpListener::bind(("::1", 0)).await?;
        let addr = listener.local_addr()?;
        let mut local = tokio::io::join(rd, wr);
        try_join!(
            async move { do_rsync(&addr, &workspace).await },
            async move {
                let (mut stream, _) = listener.accept().await?;
                tokio::io::copy_bidirectional(&mut stream, &mut local).await?;
                anyhow::Ok(())
            },
        )?;
        anyhow::Ok(())
    })
    .await
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use anyhow::anyhow;
    use tempfile::TempDir;
    use tokio::fs;
    use tokio::net::TcpListener;

    use super::*;

    #[tokio::test]
    async fn test_simple() -> Result<()> {
        let input = TempDir::new()?;
        fs::write(input.path().join("foo.txt"), "hello world").await?;

        let output = TempDir::new()?;

        let server = TcpListener::bind(("::", 0)).await?;
        let daemon = RsyncDaemon::spawn(server, output.path()).await?;
        do_rsync(daemon.addr(), input.path()).await?;
        daemon.shutdown().await?;

        assert!(!dir_diff::is_different(&input, &output).map_err(|e| anyhow!("{:?}", e))?);

        Ok(())
    }
}
