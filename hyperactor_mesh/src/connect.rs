/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::io::Cursor;
use std::pin::Pin;
use std::time::Duration;

use anyhow::Result;
use future::Future;
use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::future;
use futures::future::BoxFuture;
use futures::task::Context;
use futures::task::Poll;
use hyperactor::Actor;
use hyperactor::ActorId;
use hyperactor::Mailbox;
use hyperactor::Named;
use hyperactor::OncePortRef;
use hyperactor::PortRef;
use hyperactor::actor::RemoteActor;
use hyperactor::cap::CanSend;
use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;
use hyperactor::clock::TimeoutError;
use hyperactor::mailbox::PortReceiver;
use hyperactor::mailbox::open_once_port;
use hyperactor::mailbox::open_port;
use hyperactor::message::Bind;
use hyperactor::message::Bindings;
use hyperactor::message::Unbind;
use serde::Deserialize;
use serde::Serialize;
use tokio::io::AsyncRead;
use tokio::io::AsyncWrite;
use tokio_util::io::StreamReader;

use crate::actor_mesh::ActorMesh;

// Timeout for establishing a connection, used by both client and server.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Messages sent over the "connection" to facilitate communication.
#[derive(Debug, Serialize, Deserialize, Named, Clone)]
enum Io {
    // A data packet.
    Data(#[serde(with = "serde_bytes")] Vec<u8>),
    // Signal the end of one side of the connection.
    Eof,
}

/// A message sent from a client to initiate a connection.
#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct Connect {
    // The ID of the client that accepted the connection.
    id: ActorId,
    // The port the server can use to complete the connection.
    port: PortRef<Accept>,
}

/// A response message sent from the server back to the client to complete setting
/// up the connection.
#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct Accept {
    // The ID of the server that accepted the connection.
    id: ActorId,
    // The port the client will use to send data over the connection to the server.
    conn: PortRef<Io>,
    // Channel used by the client to send a port back to the server, which it will
    // use to send data over the connection to the client.
    return_conn: OncePortRef<PortRef<Io>>,
}

impl Bind for Connect {
    fn bind(&mut self, bindings: &mut Bindings) -> Result<()> {
        self.port.bind(bindings)
    }
}

impl Unbind for Connect {
    fn unbind(&self, bindings: &mut Bindings) -> Result<()> {
        self.port.unbind(bindings)
    }
}

struct IoMsgStream {
    port: PortReceiver<Io>,
    exhausted: bool,
}

impl Stream for IoMsgStream {
    type Item = std::io::Result<Cursor<Vec<u8>>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Once exhausted, always return None
        if self.exhausted {
            return Poll::Ready(None);
        }

        let result = futures::ready!(Box::pin(self.port.recv()).as_mut().poll(cx));
        match result {
            Err(err) => Poll::Ready(Some(Err(std::io::Error::other(err)))),
            Ok(Io::Data(buf)) => Poll::Ready(Some(Ok(Cursor::new(buf)))),
            // Break out of stream when we see EOF.
            Ok(Io::Eof) => {
                self.exhausted = true;
                Poll::Ready(None)
            }
        }
    }
}

/// Wrap a `PortReceiver<IoMsg>` as a `AsyncRead`.
pub struct IoMsgRead {
    remote: ActorId,
    inner: StreamReader<IoMsgStream, Cursor<Vec<u8>>>,
}

impl IoMsgRead {
    pub fn remote(&self) -> &ActorId {
        &self.remote
    }

    fn new(remote: ActorId, port: PortReceiver<Io>) -> Self {
        Self {
            remote,
            inner: StreamReader::new(IoMsgStream {
                port,
                exhausted: false,
            }),
        }
    }
}

impl AsyncRead for IoMsgRead {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

enum MaybeOwned<'a, T> {
    Owned(T),
    Borrowed(&'a T),
}

impl<'a, T> MaybeOwned<'a, T> {
    fn as_ref(&self) -> &T {
        match self {
            MaybeOwned::Owned(t) => t,
            MaybeOwned::Borrowed(t) => t,
        }
    }
}

impl<'a, T> std::ops::Deref for MaybeOwned<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.as_ref()
    }
}

impl<'a, T> From<&'a T> for MaybeOwned<'a, T> {
    fn from(t: &'a T) -> Self {
        MaybeOwned::Borrowed(t)
    }
}

impl<T> From<T> for MaybeOwned<'static, T> {
    fn from(t: T) -> Self {
        MaybeOwned::Owned(t)
    }
}

/// Wrap a `PortRef<IoMsg>` as a `AsyncWrite`.
pub struct IoMsgWrite<'a, C> {
    remote: ActorId,
    caps: MaybeOwned<'a, C>,
    port: PortRef<Io>,
}

impl<'a, C> IoMsgWrite<'a, C> {
    pub fn remote(&self) -> &ActorId {
        &self.remote
    }

    fn borrowed(remote: ActorId, caps: &'a C, port: PortRef<Io>) -> Self {
        Self {
            remote,
            caps: MaybeOwned::Borrowed(caps),
            port,
        }
    }

    fn owned(remote: ActorId, caps: C, port: PortRef<Io>) -> Self {
        Self {
            remote,
            caps: MaybeOwned::Owned(caps),
            port,
        }
    }
}

impl<'a, C: CanSend> AsyncWrite for IoMsgWrite<'a, C> {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, std::io::Error>> {
        match self.port.send(self.caps.as_ref(), Io::Data(buf.into())) {
            Ok(()) => Poll::Ready(Ok(buf.len())),
            Err(e) => Poll::Ready(Err(std::io::Error::other(e))),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), std::io::Error>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        // Send EOF on shutdown.
        match self.port.send(self.caps.as_ref(), Io::Eof) {
            Ok(()) => Poll::Ready(Ok(())),
            Err(e) => Poll::Ready(Err(std::io::Error::other(e))),
        }
    }
}

pub struct ConnectStream {
    mailbox: Mailbox,
    pending: usize,
    port: PortReceiver<Accept>,
    timeout: BoxFuture<'static, Result<(), TimeoutError>>,
}

impl ConnectStream {
    fn new(mailbox: Mailbox, pending: usize) -> (Connect, Self) {
        let (tx, port) = open_port::<Accept>(&mailbox);
        let connect = Connect {
            id: mailbox.actor_id().clone(),
            port: tx.bind(),
        };
        (
            connect,
            Self {
                mailbox,
                pending,
                port,
                timeout: RealClock
                    .timeout(CONNECT_TIMEOUT, futures::future::pending())
                    .boxed(),
            },
        )
    }

    pub fn for_mesh<M, A>(mesh: &M) -> (Connect, Self)
    where
        M: ActorMesh<Actor = A>,
        A: RemoteActor,
    {
        Self::new(
            mesh.proc_mesh().client().clone(),
            mesh.shape().slice().len(),
        )
    }
}

impl Stream for ConnectStream {
    type Item = Result<(IoMsgRead, IoMsgWrite<'static, Mailbox>)>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Check for stream exhaustion.
        if self.pending == 0 {
            return Poll::Ready(None);
        }

        // Check for timeout.
        if let Poll::Ready(Err(timeout)) = self.timeout.as_mut().poll(cx) {
            return Poll::Ready(Some(Err(timeout.into())));
        }

        // Poll for a new `Accept` message on the port.
        let result = futures::ready!(Box::pin(self.port.recv()).as_mut().poll(cx));
        match result {
            Ok(connection) => {
                self.pending -= 1;
                let (tx, rx) = open_port::<Io>(&self.mailbox);
                match connection.return_conn.send(&self.mailbox, tx.bind()) {
                    Ok(()) => Poll::Ready(Some(Ok((
                        IoMsgRead::new(connection.id.clone(), rx),
                        IoMsgWrite::owned(connection.id, self.mailbox.clone(), connection.conn),
                    )))),
                    Err(e) => Poll::Ready(Some(Err(e.into()))),
                }
            }
            Err(e) => Poll::Ready(Some(Err(e.into()))),
        }
    }
}

/// Helper used by `Handler<Connect>`s to accept a connection initiated by a `Connect` message and
/// return `AsyncRead` and `AsyncWrite` streams that can be used to communicate with the other side.
pub async fn accept<'a, A: Actor>(
    this: &'a hyperactor::Context<'a, A>,
    message: Connect,
) -> Result<(IoMsgRead, IoMsgWrite<'a, hyperactor::Context<'a, A>>)> {
    let (tx, rx) = open_port::<Io>(this);
    let (r_tx, r_rx) = open_once_port::<PortRef<Io>>(this);
    message.port.send(
        this,
        Accept {
            conn: tx.bind(),
            return_conn: r_tx.bind(),
            id: this.self_id().clone(),
        },
    )?;
    let wr = RealClock.timeout(CONNECT_TIMEOUT, r_rx.recv()).await??;
    Ok((
        IoMsgRead::new(message.id.clone(), rx),
        IoMsgWrite::borrowed(message.id, this, wr),
    ))
}

pub async fn connect(
    mailbox: &Mailbox,
    port: PortRef<Connect>,
) -> Result<(IoMsgRead, IoMsgWrite<Mailbox>)> {
    let (connect, mut stream) = ConnectStream::new(mailbox.clone(), 1);
    port.send(mailbox, connect)?;
    stream.next().await.expect("expected a single connection")
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use async_trait::async_trait;
    use futures::try_join;
    use hyperactor::Actor;
    use hyperactor::Context;
    use hyperactor::Handler;
    use hyperactor::proc::Proc;
    use tokio::io::AsyncReadExt;
    use tokio::io::AsyncWriteExt;

    use super::*;

    #[derive(Debug)]
    struct EchoActor {}

    #[async_trait]
    impl Actor for EchoActor {
        type Params = ();

        async fn new(_params: ()) -> Result<Self, anyhow::Error> {
            Ok(Self {})
        }
    }

    #[async_trait]
    impl Handler<Connect> for EchoActor {
        async fn handle(
            &mut self,
            this: &Context<Self>,
            message: Connect,
        ) -> Result<(), anyhow::Error> {
            let (mut rd, mut wr) = accept(this, message).await?;
            tokio::io::copy(&mut rd, &mut wr).await?;
            wr.shutdown().await?;
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_simple_connection() -> Result<()> {
        let proc = Proc::local();
        let client = proc.attach("client")?;
        let actor = proc.spawn::<EchoActor>("actor", ()).await?;
        let (mut rd, mut wr) = connect(&client, actor.port().bind()).await?;
        let send = [3u8, 4u8, 5u8, 6u8];
        try_join!(
            async move {
                wr.write_all(&send).await?;
                wr.shutdown().await?;
                anyhow::Ok(())
            },
            async {
                let mut recv = vec![];
                rd.read_to_end(&mut recv).await?;
                assert_eq!(&send, recv.as_slice());
                anyhow::Ok(())
            },
        )?;
        Ok(())
    }
}
