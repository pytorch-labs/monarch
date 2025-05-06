//! Local (in-process) channel implementation.
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::LazyLock;
use std::sync::Mutex;

use super::*;
use crate::Data;

/// Create a new local channel, returning its two ends.
pub fn new<M: RemoteMessage>() -> (impl Tx<M>, impl Rx<M>) {
    let (tx, rx) = mpsc::channel::<M>(1);
    let (mpsc_tx, status_sender) = MpscTx::new(tx, ChannelAddr::Local(0));
    let mpsc_rx = MpscRx::new(rx, ChannelAddr::Local(0), status_sender);
    (mpsc_tx, mpsc_rx)
}

// In-process channels, with a shared registry.

struct Ports {
    ports: HashMap<u64, (mpsc::Sender<Data>, watch::Receiver<TxStatus>)>,
    next_port: u64,
}

impl Ports {
    fn alloc(&mut self) -> (u64, mpsc::Receiver<Data>, watch::Sender<TxStatus>) {
        let port = self.next_port;
        self.next_port += 1;
        let (tx, rx) = mpsc::channel::<Data>(1);
        let (status_tx, status_rx) = watch::channel(TxStatus::Active);
        if self.ports.insert(port, (tx.clone(), status_rx)).is_some() {
            panic!("port reused")
        }
        (port, rx, status_tx)
    }

    fn free(&mut self, port: u64) {
        self.ports.remove(&port);
    }

    fn get(&self, port: u64) -> Option<&(mpsc::Sender<Data>, watch::Receiver<TxStatus>)> {
        self.ports.get(&port)
    }
}

impl Default for Ports {
    fn default() -> Self {
        Self {
            ports: HashMap::new(),
            next_port: 1,
        }
    }
}

static PORTS: LazyLock<Mutex<Ports>> = LazyLock::new(|| Mutex::new(Ports::default()));

#[derive(Debug)]
pub struct LocalTx<M: RemoteMessage> {
    tx: mpsc::Sender<Data>,
    port: u64,
    status: watch::Receiver<TxStatus>, // Default impl. Always reports `Active`.
    _phantom: PhantomData<M>,
}

#[async_trait]
impl<M: RemoteMessage> Tx<M> for LocalTx<M> {
    async fn post(
        &self,
        message: M,
        _return_channel: oneshot::Sender<M>,
    ) -> Result<(), SendError<M>> {
        let data: Data = match bincode::serialize(&message) {
            Ok(data) => data,
            Err(err) => return Err(SendError(err.into(), message)),
        };
        self.tx
            .send(data)
            .await
            .map_err(|_| SendError(ChannelError::Closed, message))
    }

    fn addr(&self) -> ChannelAddr {
        ChannelAddr::Local(self.port)
    }

    fn status(&self) -> &watch::Receiver<TxStatus> {
        &self.status
    }
}

#[derive(Debug)]
pub struct LocalRx<M: RemoteMessage> {
    data_rx: mpsc::Receiver<Data>,
    status_tx: watch::Sender<TxStatus>,
    port: u64,
    _phantom: PhantomData<M>,
}

#[async_trait]
impl<M: RemoteMessage> Rx<M> for LocalRx<M> {
    async fn recv(&mut self) -> Result<M, ChannelError> {
        let data = self.data_rx.recv().await.ok_or(ChannelError::Closed)?;
        bincode::deserialize(&data).map_err(ChannelError::from)
    }

    fn addr(&self) -> ChannelAddr {
        ChannelAddr::Local(self.port)
    }
}

impl<M: RemoteMessage> Drop for LocalRx<M> {
    fn drop(&mut self) {
        let _ = self.status_tx.send(TxStatus::Closed);
        PORTS.lock().unwrap().free(self.port);
    }
}

/// Dial a local port, returning a Tx for it.
pub fn dial<M: RemoteMessage>(port: u64) -> Result<LocalTx<M>, ChannelError> {
    let ports = PORTS.lock().unwrap();
    let result = ports.get(port);
    if let Some((data_tx, status_rx)) = result {
        Ok(LocalTx {
            tx: data_tx.clone(),
            port,
            status: status_rx.clone(),
            _phantom: PhantomData,
        })
    } else {
        Err(ChannelError::Closed)
    }
}

/// Serve a local port. The server is shut down when the returned Rx is dropped.
pub fn serve<M: RemoteMessage>() -> (u64, LocalRx<M>) {
    let (port, data_rx, status_tx) = PORTS.lock().unwrap().alloc();
    (
        port,
        LocalRx {
            data_rx,
            status_tx,
            port,
            _phantom: PhantomData,
        },
    )
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use super::*;

    fn unused_return_channel<M>() -> oneshot::Sender<M> {
        oneshot::channel().0
    }

    #[tokio::test]
    async fn test_local_basic() {
        let (tx, mut rx) = local::new::<u64>();

        tx.post(123, unused_return_channel()).await.unwrap();
        assert_eq!(rx.recv().await.unwrap(), 123);
    }

    #[tokio::test]
    async fn test_local_dial_serve() {
        let (port, mut rx) = local::serve::<u64>();
        assert!(port != 0);

        let tx = local::dial::<u64>(port).unwrap();

        tx.post(123, unused_return_channel()).await.unwrap();
        assert_eq!(rx.recv().await.unwrap(), 123);

        drop(rx);

        assert_matches!(
            tx.post(123, unused_return_channel()).await,
            Err(SendError(ChannelError::Closed, 123))
        );
    }

    #[tokio::test]
    async fn test_local_drop() {
        let (port, mut rx) = local::serve::<u64>();
        let tx = local::dial::<u64>(port).unwrap();

        tx.post(123, unused_return_channel()).await.unwrap();
        assert_eq!(rx.recv().await.unwrap(), 123);

        drop(rx);

        assert_matches!(
            local::dial::<u64>(port).err().unwrap(),
            ChannelError::Closed
        );
    }
}
