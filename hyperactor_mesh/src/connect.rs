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
use futures::Stream;
use futures::StreamExt;
use futures::future;
use futures::stream::FuturesUnordered;
use futures::task::Context;
use futures::task::Poll;
use hyperactor::Mailbox;
use hyperactor::Named;
use hyperactor::OncePortRef;
use hyperactor::PortRef;
use hyperactor::RemoteHandles;
use hyperactor::actor::RemoteActor;
use hyperactor::cap::CanOpenPort;
use hyperactor::cap::CanSend;
use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;
use hyperactor::mailbox::PortReceiver;
use hyperactor::mailbox::open_once_port;
use hyperactor::mailbox::open_port;
use hyperactor::message::Bind;
use hyperactor::message::Bindings;
use hyperactor::message::IndexedErasedUnbound;
use hyperactor::message::Unbind;
use ndslice::selection::dsl;
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
    // The port the server can use to complete the connection.
    port: PortRef<Accept>,
}

/// A response message sent from the server back to the client to complete setting
/// up the connection.
#[derive(Debug, Serialize, Deserialize, Named, Clone)]
pub struct Accept {
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
}

impl Stream for IoMsgStream {
    type Item = std::io::Result<Cursor<Vec<u8>>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Create a new future each time and poll it immediately
        // This works if recv() is cancellation-safe (like tokio mpsc)
        let future = async {
            match self.port.recv().await {
                Err(err) => Some(Err(std::io::Error::other(err))),
                Ok(Io::Data(buf)) => Some(Ok(Cursor::new(buf))),
                // Break out of stream when we see EOF.
                Ok(Io::Eof) => None,
            }
        };
        let mut future = Box::pin(future);
        future.as_mut().poll(cx)
    }
}

/// Wrap a `PortReceiver<IoMsg>` as a `AsyncRead`.
pub struct IoMsgRead {
    inner: StreamReader<IoMsgStream, Cursor<Vec<u8>>>,
}

impl IoMsgRead {
    fn new(port: PortReceiver<Io>) -> Self {
        Self {
            inner: StreamReader::new(IoMsgStream { port }),
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

/// Wrap a `PortRef<IoMsg>` as a `AsyncWrite`.
pub struct IoMsgWrite<'a, C: CanSend> {
    caps: &'a C,
    port: PortRef<Io>,
}

impl<'a, C: CanSend> IoMsgWrite<'a, C> {
    fn new(caps: &'a C, port: PortRef<Io>) -> Self {
        Self { caps, port }
    }
}

impl<'a, C: CanSend> AsyncWrite for IoMsgWrite<'a, C> {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, std::io::Error>> {
        match self.port.send(self.caps, Io::Data(buf.into())) {
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
        match self.port.send(self.caps, Io::Eof) {
            Ok(()) => Poll::Ready(Ok(())),
            Err(e) => Poll::Ready(Err(std::io::Error::other(e))),
        }
    }
}

/// Helper used by `Handler<Connect>`s to accept a connection initiated by a `Connect` message and
/// return `AsyncRead` and `AsyncWrite` streams that can be used to communicate with the other side.
pub async fn accept<'a, C: CanOpenPort + CanSend>(
    caps: &'a C,
    message: Connect,
) -> Result<(IoMsgRead, IoMsgWrite<'a, C>)> {
    let (tx, rx) = open_port::<Io>(caps);
    let (r_tx, r_rx) = open_once_port::<PortRef<Io>>(caps);
    message.port.send(
        caps,
        Accept {
            conn: tx.bind(),
            return_conn: r_tx.bind(),
        },
    )?;
    let wr = tokio::time::timeout(CONNECT_TIMEOUT, r_rx.recv()).await??;
    Ok((IoMsgRead::new(rx), IoMsgWrite::new(caps, wr)))
}

/// Initiate a connection to a `Handler<Connect>` and return `AsyncRead` and `AsyncWrite` streams to
/// communicate with the other side.
pub async fn connect<C: CanOpenPort + CanSend>(
    caps: &C,
    port: PortRef<Connect>,
) -> Result<(IoMsgRead, IoMsgWrite<C>)> {
    let (tx, mut rx) = open_port::<Accept>(caps);
    port.send(caps, Connect { port: tx.bind() })?;

    let connection = tokio::time::timeout(CONNECT_TIMEOUT, rx.recv()).await??;
    let (tx, rx) = open_port::<Io>(caps);
    connection.return_conn.send(caps, tx.bind())?;

    Ok((IoMsgRead::new(rx), IoMsgWrite::new(caps, connection.conn)))
}

/// Initiate connections to all ranks in a `ActorMesh<Handler<Connect>>` and run the provided
/// callback on each connection.
pub async fn connect_mesh<M, A>(
    actor_mesh: M,
    handle: impl AsyncFn(IoMsgRead, IoMsgWrite<Mailbox>) -> Result<()>,
) -> Result<()>
where
    M: ActorMesh<Actor = A>,
    A: RemoteActor + RemoteHandles<Connect> + RemoteHandles<IndexedErasedUnbound<Connect>>,
{
    let client = actor_mesh.proc_mesh().client();

    // Broadcast the initiate connection message.
    let (tx, mut rx) = client.open_port::<Accept>();
    actor_mesh.cast(dsl::all(dsl::true_()), Connect { port: tx.bind() })?;

    // Loop to process running handlers on completed connections and waiting for outstanding handlers
    // to complete.
    let mut pending = actor_mesh.shape().slice().len();
    let mut running = FuturesUnordered::default();
    let deadline = RealClock.now() + CONNECT_TIMEOUT;
    while !running.is_empty() || pending > 0 {
        tokio::select! {
            // We expect all actors to connect in the given deadline.
            res = tokio::time::timeout_at(deadline, future::pending::<()>()), if pending > 0 => res?,
            res = rx.recv() => {
                let connection = res?;
                let (tx, rx) = client.open_port::<Io>();
                connection.return_conn.send(client, tx.bind())?;
                running.push(Box::pin(handle(IoMsgRead::new(rx), IoMsgWrite::new(client, connection.conn))));
                pending -= 1;
            },
            Some(res) = running.next() => res?,
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use async_trait::async_trait;
    use futures::try_join;
    use hyperactor::Actor;
    use hyperactor::Handler;
    use hyperactor::Instance;
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
            this: &Instance<Self>,
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
