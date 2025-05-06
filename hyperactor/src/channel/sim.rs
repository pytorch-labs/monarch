// SimTx contains a way to send through the network.
// SimRx contains a way to receive messages.

//! Local simulated channel implementation.
// send leads to add to network.
use std::marker::PhantomData;
use std::sync::Arc;

use dashmap::DashMap;
use futures::executor::block_on;
use regex::Regex;
use tokio::sync::Mutex;
use tokio::sync::mpsc::Receiver;

use super::*;
use crate::PortId;
use crate::channel;
use crate::data::Serialized;
use crate::id;
use crate::mailbox::MessageEnvelope;
use crate::simnet;
use crate::simnet::Dispatcher;
use crate::simnet::Event;
use crate::simnet::OperationalMessage;
use crate::simnet::ProxyMessage;
use crate::simnet::SimNetConfig;
use crate::simnet::SimNetEdge;
use crate::simnet::SimNetError;
use crate::simnet::SimNetHandle;

lazy_static! {
    /// A handle for SimNet through which you can send and schedule events in the
    /// network.
    pub static ref HANDLE: SimNetHandle =
        simnet::start(ChannelAddr::Local(0), 1000).unwrap();
    static ref SENDER: SimDispatcher = SimDispatcher::default();
}
static SIM_LINK_BUF_SIZE: usize = 256;

/// An address for a simulated channel.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    Ord,
    PartialOrd,
    Hash
)]
pub struct SimAddr {
    /// The source address.
    src: Box<ChannelAddr>,
    /// The source proxy address.
    src_proxy: Box<ChannelAddr>,
    /// The destination address.
    dst: Box<ChannelAddr>,
    /// The destination proxy address.
    dst_proxy: Box<ChannelAddr>,
}

impl SimAddr {
    /// Creates a new SimAddr.
    pub fn new(
        src: ChannelAddr,
        src_proxy: ChannelAddr,
        dst: ChannelAddr,
        dst_proxy: ChannelAddr,
    ) -> Result<Self, SimNetError> {
        if let ChannelAddr::Sim(_) = &src_proxy {
            return Err(SimNetError::InvalidArg(format!(
                "src cannot be a sim address, found {}",
                src_proxy
            )));
        }
        if let ChannelAddr::Sim(_) = &dst_proxy {
            return Err(SimNetError::InvalidArg(format!(
                "dst cannot be a sim address, found {}",
                src_proxy
            )));
        }
        if src_proxy.transport() != dst_proxy.transport() {
            return Err(SimNetError::InvalidArg(format!(
                "src_proxy and dst_proxy must have the same transport, found src_proxy: {}, dst_proxy: {}",
                src_proxy, dst_proxy
            )));
        }
        Ok(Self {
            src: Box::new(src),
            src_proxy: Box::new(src_proxy),
            dst: Box::new(dst),
            dst_proxy: Box::new(dst_proxy),
        })
    }

    /// Returns the source address.
    pub fn src(&self) -> &ChannelAddr {
        &self.src
    }

    /// Returns the source proxy address.
    pub fn src_proxy(&self) -> &ChannelAddr {
        &self.src_proxy
    }

    /// Returns the destination address.
    pub fn dst(&self) -> &ChannelAddr {
        &self.dst
    }

    /// Returns the destination proxy address.
    pub fn dst_proxy(&self) -> &ChannelAddr {
        &self.dst_proxy
    }

    /// Returns the reversed address (src <-> dst, src_proxy <-> dst_proxy).
    pub fn reversed(&self) -> Self {
        Self {
            src: self.dst.clone(),
            src_proxy: self.dst_proxy.clone(),
            dst: self.src.clone(),
            dst_proxy: self.src_proxy.clone(),
        }
    }
}

/// Configuration for an overlay edge in the simulated network.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    Ord,
    PartialOrd,
    Hash
)]
pub struct OverlayEdgeConfig {
    /// The overlay edge that specifies the source and destination proxies.
    overlay_edge: OverlayEdge,
}

/// An overlay edge in the simulated network, it specifies the src and dst proxy addresses.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    Ord,
    PartialOrd,
    Hash
)]
pub struct OverlayEdge {
    /// The source node proxy address.
    src: ChannelAddr,
    /// The destination node proxy address.
    dst: ChannelAddr,
}

impl OverlayEdgeConfig {
    pub(crate) fn new(sim_addr: SimAddr) -> Self {
        Self {
            overlay_edge: OverlayEdge {
                src: *sim_addr.src_proxy.clone(),
                dst: *sim_addr.dst_proxy.clone(),
            },
        }
    }

    pub(crate) fn src(&self) -> &ChannelAddr {
        &self.overlay_edge.src
    }

    pub(crate) fn dst(&self) -> &ChannelAddr {
        &self.overlay_edge.dst
    }
}

impl fmt::Display for SimAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{},{},{},{}",
            self.src, self.src_proxy, self.dst, self.dst_proxy
        )
    }
}

/// Message Event that can be passed around in the simnet.
#[derive(Debug)]
pub(crate) struct MessageDeliveryEvent {
    dest_addr: SimAddr,
    data: Serialized,
    duration_ms: u64,
}

impl MessageDeliveryEvent {
    /// Creates a new MessageDeliveryEvent.
    pub fn new(dest_addr: SimAddr, data: Serialized) -> Self {
        Self {
            dest_addr,
            data,
            duration_ms: 1,
        }
    }
}

#[async_trait]
impl Event for MessageDeliveryEvent {
    async fn handle(&self) -> Result<(), SimNetError> {
        // Send the message to the correct receiver.
        SENDER
            .send(self.dest_addr.clone(), self.data.clone())
            .await?;
        Ok(())
    }

    fn duration_ms(&self) -> u64 {
        self.duration_ms
    }

    fn summary(&self) -> String {
        format!(
            "Sending message from {} to {}",
            self.dest_addr.src.clone(),
            self.dest_addr.dst.clone()
        )
    }

    async fn read_simnet_config(&mut self, topology: &Arc<Mutex<SimNetConfig>>) {
        let edge = SimNetEdge {
            src: *self.dest_addr.src.clone(),
            dst: *self.dest_addr.dst.clone(),
        };
        self.duration_ms = topology
            .lock()
            .await
            .topology
            .get(&edge)
            .map_or_else(|| 1, |v| v.latency.as_millis() as u64);
    }
}

/// Export the message delivery records of the simnet.
pub async fn records() -> Option<Vec<simnet::SimulatorEventRecord>> {
    HANDLE.records().await
}

/// Bind a channel address to the simnet. It will register the address as a node in simnet,
/// and configure default latencies between this node and all other existing nodes.
pub async fn bind(addr: ChannelAddr) -> anyhow::Result<(), SimNetError> {
    HANDLE.bind(addr).await
}

/// Update the configuration for simnet.
pub async fn update_config(config: simnet::NetworkConfig) -> anyhow::Result<(), SimNetError> {
    // Only update network config for now, will add host config in the future.
    HANDLE.update_network_config(config).await
}

/// Adds a proxy to simnet so it can communicate with external nodes.
pub async fn add_proxy(addr: ChannelAddr) -> anyhow::Result<(), SimNetError> {
    HANDLE.add_proxy(addr).await
}

/// Moves the operational message receiver out of the simnet.
pub async fn operational_message_receiver()
-> anyhow::Result<Receiver<OperationalMessage>, SimNetError> {
    HANDLE.operational_message_receiver().await
}

/// Returns a simulated channel address that is bound to "any" channel address.
pub(crate) fn any(overlay_edge_config: OverlayEdgeConfig) -> ChannelAddr {
    let src_proxy = overlay_edge_config.src();
    let dst_proxy = overlay_edge_config.dst();
    ChannelAddr::Sim(SimAddr {
        src: Box::new(ChannelAddr::any(src_proxy.transport().clone())),
        src_proxy: Box::new(src_proxy.clone()),
        dst: Box::new(ChannelAddr::any(dst_proxy.transport().clone())),
        dst_proxy: Box::new(dst_proxy.clone()),
    })
}

/// Parse the sim channel address. It should have two non-sim channel addresses separated by a comma.
pub fn parse(addr_string: &str) -> Result<ChannelAddr, ChannelError> {
    let re = Regex::new(r"^([^,]+),([^,]+),([^,]+),([^,]+)$").map_err(|err| {
        ChannelError::InvalidAddress(format!("invalid sim address regex: {}", err))
    })?;

    let result = re.captures(addr_string);
    if let Some(caps) = result {
        let src_str = caps.get(1).map_or("", |m| m.as_str());
        let src_proxy_str = caps.get(2).map_or("", |m| m.as_str());
        let dst_str = caps.get(3).map_or("", |m| m.as_str());
        let dst_proxy_str = caps.get(4).map_or("", |m| m.as_str());

        if src_str.starts_with("sim!")
            || src_proxy_str.starts_with("sim!")
            || dst_str.starts_with("sim!")
            || dst_proxy_str.starts_with("sim")
        {
            return Err(ChannelError::InvalidAddress(addr_string.to_string()));
        }

        let src = src_str.parse::<ChannelAddr>()?;
        let src_proxy = src_proxy_str.parse::<ChannelAddr>()?;
        let dst = dst_str.parse::<ChannelAddr>()?;
        let dst_proxy = dst_proxy_str.parse::<ChannelAddr>()?;

        Ok(ChannelAddr::Sim(SimAddr::new(
            src, src_proxy, dst, dst_proxy,
        )?))
    } else {
        Err(ChannelError::InvalidAddress(addr_string.to_string()))
    }
}

impl<M: RemoteMessage> Drop for SimRx<M> {
    fn drop(&mut self) {
        // Remove the sender from the dispatchers.
        SENDER.dispatchers.remove(&self.addr);
    }
}

/// Primarily used for dispatching messages to the correct sender.
#[derive(Debug)]
pub struct SimDispatcher {
    dispatchers: DashMap<ChannelAddr, mpsc::Sender<Serialized>>,
    sender_cache: DashMap<ChannelAddr, Arc<dyn Tx<MessageEnvelope> + Send + Sync>>,
}

fn create_egress_sender(
    addr: ChannelAddr,
) -> anyhow::Result<Arc<dyn Tx<MessageEnvelope> + Send + Sync>> {
    let tx = channel::dial(addr)?;
    Ok(Arc::new(tx))
}

/// Check if the address is outside of the simulation.
pub async fn is_external_addr(addr: &SimAddr) -> bool {
    let result = HANDLE.proxy_addr().await;
    if let Some(local_proxy_addr) = result {
        addr.dst_proxy() != &local_proxy_addr
    } else {
        // If there's no local proxy, the dst proxy is different from local proxy.
        true
    }
}

#[async_trait]
impl Dispatcher<SimAddr> for SimDispatcher {
    async fn send(&self, addr: SimAddr, data: Serialized) -> Result<(), SimNetError> {
        if is_external_addr(&addr).await {
            let dst_proxy = *addr.dst_proxy.clone();
            let sender = self
                .sender_cache
                .entry(dst_proxy.clone())
                .or_insert_with(|| create_egress_sender(dst_proxy.clone()).unwrap());
            let forward_message = ProxyMessage::new(Some(addr.clone()), data);
            let serialized_forward_message = match Serialized::serialize(&forward_message) {
                Ok(data) => data,
                Err(err) => return Err(SimNetError::InvalidArg(err.to_string())),
            };
            // Here we use mailbox to deliver the ForwardMessage. But it's higher level than
            // the simnet. So there are unused placeholder here which is not ideal.
            let port_id_placeholder = PortId(id!(unused_world[0].unused_actor), 0);
            let message =
                MessageEnvelope::new_unknown(port_id_placeholder, serialized_forward_message);
            return sender
                .post(message, oneshot::channel().0)
                .await
                .map_err(|err| SimNetError::InvalidNode(dst_proxy.to_string(), err.into()));
        }

        let dst = *addr.dst.clone();
        self.dispatchers
            .get(&dst)
            .ok_or_else(|| {
                SimNetError::InvalidNode(dst.to_string(), anyhow::anyhow!("no dispatcher found"))
            })?
            .send(data)
            .await
            .map_err(|err| SimNetError::InvalidNode(dst.to_string(), err.into()))
    }
}

impl Default for SimDispatcher {
    fn default() -> Self {
        Self {
            dispatchers: DashMap::new(),
            sender_cache: DashMap::new(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct SimTx<M: RemoteMessage> {
    addr: SimAddr,
    status: watch::Receiver<TxStatus>, // Default impl. Always reports `Active`.
    _phantom: PhantomData<M>,
}

#[derive(Debug)]
pub(crate) struct SimRx<M: RemoteMessage> {
    /// The destination address, not the full SimAddr.
    addr: ChannelAddr,
    rx: mpsc::Receiver<Serialized>,
    _phantom: PhantomData<M>,
}

#[async_trait]
impl<M: RemoteMessage> Tx<M> for SimTx<M> {
    async fn post(
        &self,
        message: M,
        _return_handle: oneshot::Sender<M>,
    ) -> Result<(), SendError<M>> {
        let data = match Serialized::serialize(&message) {
            Ok(data) => data,
            Err(err) => return Err(SendError(err.into(), message)),
        };
        HANDLE
            .send_event(Box::new(MessageDeliveryEvent::new(self.addr.clone(), data)))
            .await
            .map_err(|err| SendError(ChannelError::from(err), message))
    }

    fn addr(&self) -> ChannelAddr {
        ChannelAddr::Sim(self.addr.clone())
    }

    fn status(&self) -> &watch::Receiver<TxStatus> {
        &self.status
    }
}

/// Dial a peer and return a transmitter. The transmitter can retrieve from the
/// network the link latency.
pub(crate) fn dial<M: RemoteMessage>(sim_addr: SimAddr) -> Result<SimTx<M>, ChannelError> {
    // This watch channel always reports active. The sender is
    // dropped.
    let (_, status) = watch::channel(TxStatus::Active);
    Ok(SimTx {
        addr: sim_addr,
        status,
        _phantom: PhantomData,
    })
}

/// Serve a sim channel. Set up the right simulated sender and receivers
/// The mpsc tx will be used to dispatch messages when it's time while
/// the mpsc rx will be used by the above applications to handle received messages
/// like any other channel.
/// A sim address has src and dst. Dispatchers are only indexed by dst address.
pub(crate) fn serve<M: RemoteMessage>(
    sim_addr: SimAddr,
) -> anyhow::Result<(ChannelAddr, SimRx<M>)> {
    // Serves sim address at sim_addr.src and set up local proxy at sim_addr.src_proxy.
    // Reversing the src and dst since the first element in the output tuple is the
    // dialing address of this sim channel. So the served address is the dst.
    let sim_addr = sim_addr.reversed();
    tracing::info!("adding proxy for sim addr: {:#?}", &sim_addr);
    block_on(add_proxy(sim_addr.dst_proxy().clone()))?;
    let (tx, rx) = mpsc::channel::<Serialized>(SIM_LINK_BUF_SIZE);
    // Add tx to sender dispatch.
    SENDER.dispatchers.insert(sim_addr.dst().clone(), tx);
    // Return the sender.
    Ok((
        ChannelAddr::Sim(sim_addr.clone()),
        SimRx {
            addr: sim_addr.dst().clone(),
            rx,
            _phantom: PhantomData,
        },
    ))
}

#[async_trait]
impl<M: RemoteMessage> Rx<M> for SimRx<M> {
    async fn recv(&mut self) -> Result<M, ChannelError> {
        let data = self.rx.recv().await.ok_or(ChannelError::Closed)?;
        data.deserialized().map_err(ChannelError::from)
    }

    fn addr(&self) -> ChannelAddr {
        self.addr.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use super::*;
    use crate::PortId;
    use crate::id;

    #[tokio::test]
    async fn test_sim_basic() {
        let dst_ok = vec!["[::1]:1234", "tcp!127.0.0.1:8080", "local!123"];
        let srcs_ok = vec!["[::2]:1234", "tcp!127.0.0.2:8080", "local!124"];

        // TODO: New NodeAdd event should do this for you..
        for addr in dst_ok.iter().chain(srcs_ok.iter()) {
            // Add to network along with its edges.
            sim::HANDLE
                .bind(addr.parse::<ChannelAddr>().unwrap())
                .await
                .unwrap();
        }
        // Messages are transferred internally if only there's a local proxy and the
        // dst proxy is the same as local proxy.
        let proxy = ChannelAddr::any(ChannelTransport::Unix);
        for (src_addr, dst_addr) in zip(srcs_ok, dst_ok) {
            let channel_addr = ChannelAddr::Sim(
                SimAddr::new(
                    src_addr.parse::<ChannelAddr>().unwrap(),
                    proxy.clone(),
                    dst_addr.parse::<ChannelAddr>().unwrap(),
                    proxy.clone(),
                )
                .unwrap(),
            );
            let ChannelAddr::Sim(sim_addr) = channel_addr.clone() else {
                panic!("expected sim");
            };

            // reverse src and dst since `sim::serve()` will reverse it.
            let (_, mut rx) = sim::serve::<u64>(sim_addr.reversed().clone()).unwrap();
            let tx = sim::dial::<u64>(sim_addr).unwrap();
            tx.post(123, oneshot::channel().0).await.unwrap();
            assert_eq!(rx.recv().await.unwrap(), 123);
        }

        let records = sim::records().await;
        eprintln!("records: {:#?}", records);
    }

    #[tokio::test]
    async fn test_send_egress_message() {
        // Serve an external proxy channel to receive the egress message.
        let egress_addr = ChannelAddr::any(ChannelTransport::Unix);
        let dispatcher = SimDispatcher::default();
        let (_, mut rx) = channel::serve::<MessageEnvelope>(egress_addr.clone())
            .await
            .unwrap();
        // just a random port ID
        let port_id = PortId(id!(test[0].actor0), 0);
        let msg = MessageEnvelope::new_unknown(
            port_id.clone(),
            Serialized::serialize(&"hola".to_string()).unwrap(),
        );
        // The sim addr we want simnet to send message to, it should have the egress_addr
        // as the proxy address of dst.
        let sim_addr = SimAddr::new(
            "unix!@src".parse::<ChannelAddr>().unwrap(),
            "unix!@src_proxy".parse::<ChannelAddr>().unwrap(),
            "unix!@dst".parse::<ChannelAddr>().unwrap(),
            egress_addr,
        )
        .unwrap();
        let serialized_msg = Serialized::serialize(&msg).unwrap();
        dispatcher
            .send(sim_addr.clone(), serialized_msg.clone())
            .await
            .unwrap();
        let received_msg = rx.recv().await.unwrap();
        let actual_forward_msg: ProxyMessage = received_msg.deserialized().unwrap();
        let expected_forward_msg = ProxyMessage::new(Some(sim_addr.clone()), serialized_msg);

        assert_eq!(actual_forward_msg, expected_forward_msg);

        // Sending the message again should work by using the cached sender.
        // But it's impl detail, not verified here. We just verify that it
        // can send a different message.
        let msg = MessageEnvelope::new_unknown(
            port_id,
            Serialized::serialize(&"ciao".to_string()).unwrap(),
        );
        let serialized_msg = Serialized::serialize(&msg).unwrap();
        dispatcher
            .send(sim_addr.clone(), serialized_msg.clone())
            .await
            .unwrap();
        let received_msg = rx.recv().await.unwrap();
        let actual_forward_msg: ProxyMessage = received_msg.deserialized().unwrap();
        let expected_forward_msg = ProxyMessage::new(Some(sim_addr.clone()), serialized_msg);
        assert_eq!(actual_forward_msg, expected_forward_msg);
    }

    #[tokio::test]
    async fn test_invalid_sim_addr() {
        let src = "sim!src";
        let dst = "sim!dst";
        let src_proxy = "sim!src_proxy";
        let dst_proxy = "sim!dst_proxy";
        let sim_addr = format!("{},{},{},{}", src, src_proxy, dst, dst_proxy);
        let result = parse(&sim_addr);
        assert!(matches!(result, Err(ChannelError::InvalidAddress(_))));

        let src = "unix!src".parse::<ChannelAddr>().unwrap();
        let dst = "unix!dst".parse::<ChannelAddr>().unwrap();
        let src_proxy = "unix!src_proxy".parse::<ChannelAddr>().unwrap();
        let dst_proxy = "sim!unix!a,unix!b,unix!c,unix!d"
            .parse::<ChannelAddr>()
            .unwrap();
        let result = SimAddr::new(src.clone(), src_proxy, dst.clone(), dst_proxy);
        // dst_proxy shouldn't be a sim address.
        assert!(matches!(result, Err(SimNetError::InvalidArg(_))));

        // src_proxy and dst_proxy should have the same transport.
        let src_proxy = "tcp![::]:1".parse::<ChannelAddr>().unwrap();
        let dst_proxy = "unix!dst_proxy".parse::<ChannelAddr>().unwrap();
        let result = SimAddr::new(src, src_proxy, dst, dst_proxy);
        assert!(matches!(result, Err(SimNetError::InvalidArg(_))));
    }
}
