/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
use tokio::sync::mpsc::UnboundedReceiver;

use super::*;
use crate::PortId;
use crate::channel;
use crate::clock::Clock;
use crate::clock::RealClock;
use crate::clock::SimClock;
use crate::data::Serialized;
use crate::id;
use crate::mailbox::MessageEnvelope;
use crate::simnet;
use crate::simnet::Dispatcher;
use crate::simnet::Event;
use crate::simnet::OperationalMessage;
use crate::simnet::ProxyMessage;
use crate::simnet::ScheduledEvent;
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
static CLIENT_ADDRESS: &str = "unix!@client";

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
/// A channel address along with the address of the proxy for the process
pub struct AddressProxyPair {
    /// The address.
    pub address: ChannelAddr,
    /// The address of the proxy for the process
    pub proxy: ChannelAddr,
}

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
    src: Option<Box<AddressProxyPair>>,
    /// The address.
    addr: Box<ChannelAddr>,
    /// The proxy address.
    proxy: Box<ChannelAddr>,
}

impl SimAddr {
    /// Creates a new SimAddr.
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `SimNetError`.
    /// Creates a new SimAddr without a source to be served
    pub fn new(addr: ChannelAddr, proxy: ChannelAddr) -> Result<Self, SimNetError> {
        Self::new_impl(None, addr, proxy)
    }

    /// Creates a new directional SimAddr meant to convey a channel between two addresses.
    pub fn new_with_src(
        src: AddressProxyPair,
        addr: ChannelAddr,
        proxy: ChannelAddr,
    ) -> Result<Self, SimNetError> {
        Self::new_impl(Some(Box::new(src)), addr, proxy)
    }

    fn new_impl(
        src: Option<Box<AddressProxyPair>>,
        addr: ChannelAddr,
        proxy: ChannelAddr,
    ) -> Result<Self, SimNetError> {
        if let ChannelAddr::Sim(_) = &addr {
            return Err(SimNetError::InvalidArg(format!(
                "addr cannot be a sim address, found {}",
                addr
            )));
        }
        if let ChannelAddr::Sim(_) = &proxy {
            return Err(SimNetError::InvalidArg(format!(
                "proxy cannot be a sim address, found {}",
                proxy
            )));
        }
        Ok(Self {
            src,
            addr: Box::new(addr),
            proxy: Box::new(proxy),
        })
    }

    /// Returns the address.
    pub fn addr(&self) -> &ChannelAddr {
        &self.addr
    }

    /// Returns the proxy address.
    pub fn proxy(&self) -> &ChannelAddr {
        &self.proxy
    }

    /// Returns the source address and proxy.
    pub fn src(&self) -> &Option<Box<AddressProxyPair>> {
        &self.src
    }
}

impl fmt::Display for SimAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.src {
            None => write!(f, "{},{}", self.addr, self.proxy),
            Some(src) => write!(
                f,
                "{},{},{},{}",
                src.address, src.proxy, self.addr, self.proxy
            ),
        }
    }
}

/// Message Event that can be passed around in the simnet.
#[derive(Debug)]
pub(crate) struct MessageDeliveryEvent {
    src_addr: Option<AddressProxyPair>,
    dest_addr: AddressProxyPair,
    data: Serialized,
    duration_ms: u64,
}

impl MessageDeliveryEvent {
    /// Creates a new MessageDeliveryEvent.
    pub fn new(
        src_addr: Option<AddressProxyPair>,
        dest_addr: AddressProxyPair,
        data: Serialized,
    ) -> Self {
        Self {
            src_addr,
            dest_addr,
            data,
            duration_ms: 100,
        }
    }
}

#[async_trait]
impl Event for MessageDeliveryEvent {
    async fn handle(&self) -> Result<(), SimNetError> {
        // Send the message to the correct receiver.
        SENDER
            .send(
                self.src_addr.clone(),
                self.dest_addr.clone(),
                self.data.clone(),
            )
            .await?;
        Ok(())
    }

    fn duration_ms(&self) -> u64 {
        self.duration_ms
    }

    fn summary(&self) -> String {
        format!(
            "Sending message from {} to {}",
            self.src_addr
                .as_ref()
                .map_or("unknown".to_string(), |addr| addr.address.to_string()),
            self.dest_addr.address.clone()
        )
    }

    async fn read_simnet_config(&mut self, topology: &Arc<Mutex<SimNetConfig>>) {
        if let Some(src_addr) = &self.src_addr {
            let edge = SimNetEdge {
                src: src_addr.address.clone(),
                dst: self.dest_addr.address.clone(),
            };
            self.duration_ms = topology
                .lock()
                .await
                .topology
                .get(&edge)
                .map_or_else(|| 1, |v| v.latency.as_millis() as u64);
        }
    }
}

/// Export the message delivery records of the simnet.
pub async fn records() -> Option<Vec<simnet::SimulatorEventRecord>> {
    HANDLE.records().await
}

/// Bind a channel address to the simnet. It will register the address as a node in simnet,
/// and configure default latencies between this node and all other existing nodes.
pub async fn bind(addr: ChannelAddr) -> anyhow::Result<(), SimNetError> {
    HANDLE.bind(addr)
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
-> anyhow::Result<UnboundedReceiver<OperationalMessage>, SimNetError> {
    HANDLE.operational_message_receiver().await
}

/// Returns a simulated channel address that is bound to "any" channel address.
pub(crate) fn any(proxy: ChannelAddr) -> ChannelAddr {
    ChannelAddr::Sim(SimAddr {
        src: None,
        addr: Box::new(ChannelAddr::any(proxy.transport())),
        proxy: Box::new(proxy),
    })
}

/// Parse the sim channel address. It should have two non-sim channel addresses separated by a comma.
#[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `ChannelError`.
pub fn parse(addr_string: &str) -> Result<ChannelAddr, ChannelError> {
    let re = Regex::new(r"([^,]+),([^,]+)(,([^,]+),([^,]+))?$").map_err(|err| {
        ChannelError::InvalidAddress(format!("invalid sim address regex: {}", err))
    })?;

    let result = re.captures(addr_string);
    if let Some(caps) = result {
        let parts = caps
            .iter()
            .skip(1)
            .map(|cap| cap.map_or("", |m| m.as_str()))
            .filter(|m| !m.is_empty())
            .collect::<Vec<_>>();

        if parts.iter().any(|part| part.starts_with("sim!")) {
            return Err(ChannelError::InvalidAddress(addr_string.to_string()));
        }

        match parts.len() {
            2 => {
                let addr = parts[0].parse::<ChannelAddr>()?;
                let proxy = parts[1].parse::<ChannelAddr>()?;

                Ok(ChannelAddr::Sim(SimAddr::new(addr, proxy)?))
            }
            5 => {
                let src_addr = parts[0].parse::<ChannelAddr>()?;
                let src_proxy = parts[1].parse::<ChannelAddr>()?;
                let addr = parts[3].parse::<ChannelAddr>()?;
                let proxy = parts[4].parse::<ChannelAddr>()?;

                Ok(ChannelAddr::Sim(SimAddr::new_with_src(
                    AddressProxyPair {
                        address: src_addr,
                        proxy: src_proxy,
                    },
                    addr,
                    proxy,
                )?))
            }
            _ => Err(ChannelError::InvalidAddress(addr_string.to_string())),
        }
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
pub async fn is_external_addr(addr: &AddressProxyPair) -> bool {
    HANDLE
        .proxy_addr()
        .await
        .is_none_or(|local_proxy| local_proxy != addr.proxy)
}

#[async_trait]
impl Dispatcher<AddressProxyPair> for SimDispatcher {
    async fn send(
        &self,
        src_addr: Option<AddressProxyPair>,
        addr: AddressProxyPair,
        data: Serialized,
    ) -> Result<(), SimNetError> {
        if is_external_addr(&addr).await {
            let dst_proxy = addr.proxy.clone();
            let sender = self
                .sender_cache
                .entry(dst_proxy.clone())
                .or_insert_with(|| create_egress_sender(dst_proxy.clone()).unwrap());
            let forward_message = ProxyMessage::new(src_addr.clone(), Some(addr.clone()), data);
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
                .try_post(message, oneshot::channel().0)
                .map_err(|err| SimNetError::InvalidNode(addr.address.to_string(), err.into()));
        }

        self.dispatchers
            .get(&addr.address)
            .ok_or_else(|| {
                SimNetError::InvalidNode(
                    addr.address.to_string(),
                    anyhow::anyhow!("no dispatcher found"),
                )
            })?
            .send(data)
            .await
            .map_err(|err| SimNetError::InvalidNode(addr.address.to_string(), err.into()))
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
    src_addr: Option<AddressProxyPair>,
    dst_addr: AddressProxyPair,
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
    fn try_post(&self, message: M, _return_handle: oneshot::Sender<M>) -> Result<(), SendError<M>> {
        let data = match Serialized::serialize(&message) {
            Ok(data) => data,
            Err(err) => return Err(SendError(err.into(), message)),
        };
        match &self.src_addr {
            Some(src_addr) if src_addr.address.to_string() == CLIENT_ADDRESS => HANDLE
                .send_scheduled_event(ScheduledEvent {
                    event: Box::new(MessageDeliveryEvent::new(
                        self.src_addr.clone(),
                        self.dst_addr.clone(),
                        data,
                    )),
                    time: SimClock.millis_since_start(RealClock.now()),
                })
                .map_err(|err: SimNetError| SendError(ChannelError::from(err), message)),
            _ => HANDLE
                .send_event(Box::new(MessageDeliveryEvent::new(
                    self.src_addr.clone(),
                    self.dst_addr.clone(),
                    data,
                )))
                .map_err(|err| SendError(ChannelError::from(err), message)),
        }
    }

    fn addr(&self) -> ChannelAddr {
        self.dst_addr.address.clone()
    }

    fn status(&self) -> &watch::Receiver<TxStatus> {
        &self.status
    }
}

/// Dial a peer and return a transmitter. The transmitter can retrieve from the
/// network the link latency.
#[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `ChannelError`.
pub(crate) fn dial<M: RemoteMessage>(addr: SimAddr) -> Result<SimTx<M>, ChannelError> {
    // This watch channel always reports active. The sender is
    // dropped.
    let (_, status) = watch::channel(TxStatus::Active);
    let dialer = addr.src().clone().map(|src| *src);

    Ok(SimTx {
        src_addr: dialer,
        dst_addr: AddressProxyPair {
            address: *addr.addr,
            proxy: *addr.proxy,
        },
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
    tracing::info!("adding proxy for sim addr: {:#?}", &sim_addr);
    block_on(add_proxy(*sim_addr.proxy.clone()))?;
    let (tx, rx) = mpsc::channel::<Serialized>(SIM_LINK_BUF_SIZE);
    // Add tx to sender dispatch.
    SENDER.dispatchers.insert(*sim_addr.addr.clone(), tx);
    // Return the sender.
    Ok((
        ChannelAddr::Sim(sim_addr.clone()),
        SimRx {
            addr: *sim_addr.addr.clone(),
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
    use crate::clock::Clock;
    use crate::clock::RealClock;
    use crate::clock::SimClock;
    use crate::simnet::NetworkConfig;

    #[tokio::test]
    async fn test_sim_basic() {
        let dst_ok = vec!["[::1]:1234", "tcp!127.0.0.1:8080", "local!123"];
        let srcs_ok = vec!["[::2]:1234", "tcp!127.0.0.2:8080", "local!124"];

        // TODO: New NodeAdd event should do this for you..
        for addr in dst_ok.iter().chain(srcs_ok.iter()) {
            // Add to network along with its edges.
            sim::HANDLE
                .bind(addr.parse::<ChannelAddr>().unwrap())
                .unwrap();
        }
        // Messages are transferred internally if only there's a local proxy and the
        // dst proxy is the same as local proxy.
        let proxy = ChannelAddr::any(ChannelTransport::Unix);
        for (src_addr, dst_addr) in zip(srcs_ok, dst_ok) {
            let dst_addr = SimAddr::new_with_src(
                AddressProxyPair {
                    address: src_addr.parse::<ChannelAddr>().unwrap(),
                    proxy: proxy.clone(),
                },
                dst_addr.parse::<ChannelAddr>().unwrap(),
                proxy.clone(),
            )
            .unwrap();

            let (_, mut rx) = sim::serve::<u64>(dst_addr.clone()).unwrap();
            let tx = sim::dial::<u64>(dst_addr).unwrap();
            tx.try_post(123, oneshot::channel().0).unwrap();
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
        let src_addr = AddressProxyPair {
            address: "unix!@src".parse::<ChannelAddr>().unwrap(),
            proxy: "unix!@proxy".parse::<ChannelAddr>().unwrap(),
        };
        let egress_addr = AddressProxyPair {
            address: "unix!@dst".parse::<ChannelAddr>().unwrap(),
            proxy: egress_addr,
        };
        let serialized_msg = Serialized::serialize(&msg).unwrap();
        dispatcher
            .send(
                Some(src_addr.clone()),
                egress_addr.clone(),
                serialized_msg.clone(),
            )
            .await
            .unwrap();
        let received_msg = rx.recv().await.unwrap();
        let actual_forward_msg: ProxyMessage = received_msg.deserialized().unwrap();
        let expected_forward_msg = ProxyMessage::new(
            Some(src_addr.clone()),
            Some(egress_addr.clone()),
            serialized_msg,
        );

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
            .send(
                Some(src_addr.clone()),
                egress_addr.clone(),
                serialized_msg.clone(),
            )
            .await
            .unwrap();
        let received_msg = rx.recv().await.unwrap();
        let actual_forward_msg: ProxyMessage = received_msg.deserialized().unwrap();
        let expected_forward_msg =
            ProxyMessage::new(Some(src_addr), Some(egress_addr), serialized_msg);
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

        let dst = "unix!dst".parse::<ChannelAddr>().unwrap();
        let dst_proxy = "sim!unix!a,unix!b".parse::<ChannelAddr>().unwrap();
        let result = SimAddr::new(dst, dst_proxy);
        // dst_proxy shouldn't be a sim address.
        assert!(matches!(result, Err(SimNetError::InvalidArg(_))));
    }

    #[tokio::test]
    async fn test_parse_sim_addr() {
        let sim_addr = "sim!unix!@dst,unix!@proxy";
        let result = sim_addr.parse();
        assert!(result.is_ok());
        let ChannelAddr::Sim(sim_addr) = result.unwrap() else {
            panic!("Expected a sim address");
        };
        assert!(sim_addr.src().is_none());
        assert_eq!(sim_addr.addr().to_string(), "unix!@dst");
        assert_eq!(sim_addr.proxy().to_string(), "unix!@proxy");

        let sim_addr = "sim!unix!@src,unix!@proxy,unix!@dst,unix!@proxy";
        let result = sim_addr.parse();
        assert!(result.is_ok());
        let ChannelAddr::Sim(sim_addr) = result.unwrap() else {
            panic!("Expected a sim address");
        };
        assert!(sim_addr.src().is_some());
        let src_pair = sim_addr.src().clone().unwrap();
        assert_eq!(src_pair.address.to_string(), "unix!@src");
        assert_eq!(src_pair.proxy.to_string(), "unix!@proxy");
        assert_eq!(sim_addr.addr().to_string(), "unix!@dst");
        assert_eq!(sim_addr.proxy().to_string(), "unix!@proxy");
    }

    #[tokio::test]
    async fn test_realtime_frontier() {
        tokio::time::pause();
        let sim_addr = SimAddr::new(
            "unix!@dst".parse::<ChannelAddr>().unwrap(),
            "unix!@proxy".parse::<ChannelAddr>().unwrap(),
        )
        .unwrap();
        let sim_addr_with_src = SimAddr::new_with_src(
            AddressProxyPair {
                address: "unix!@src".parse::<ChannelAddr>().unwrap(),
                proxy: "unix!@proxy".parse::<ChannelAddr>().unwrap(),
            },
            "unix!@dst".parse::<ChannelAddr>().unwrap(),
            "unix!@proxy".parse::<ChannelAddr>().unwrap(),
        )
        .unwrap();
        let (_, mut rx) = sim::serve::<()>(sim_addr.clone()).unwrap();
        let tx = sim::dial::<()>(sim_addr_with_src).unwrap();
        let simnet_config_yaml = r#"
        edges:
        - src: unix!@src
          dst: unix!@dst
          metadata:
            latency: 100
        "#;
        update_config(NetworkConfig::from_yaml(simnet_config_yaml).unwrap())
            .await
            .unwrap();

        // This message will be delievered at simulator time = 100 seconds
        tx.try_post((), oneshot::channel().0).unwrap();
        {
            // Allow some time for simnet to run
            RealClock.sleep(tokio::time::Duration::from_secs(1)).await;
            // Messages have not been receive since 10 seconds have not elapsed
            assert!(rx.rx.try_recv().is_err());
        }
        // Advance "real" time by 100 seconds
        tokio::time::advance(tokio::time::Duration::from_secs(100)).await;
        {
            // Allow some time for simnet to run
            RealClock.sleep(tokio::time::Duration::from_secs(1)).await;
            // Messages are received
            assert!(rx.rx.try_recv().is_ok());
        }
    }

    #[tokio::test]
    async fn test_client_message_scheduled_realtime() {
        tokio::time::pause();
        let controller_to_dst = SimAddr::new_with_src(
            AddressProxyPair {
                address: "unix!@controller".parse::<ChannelAddr>().unwrap(),
                proxy: "unix!@proxy".parse::<ChannelAddr>().unwrap(),
            },
            "unix!@dst".parse::<ChannelAddr>().unwrap(),
            "unix!@proxy".parse::<ChannelAddr>().unwrap(),
        )
        .unwrap();
        let controller_tx = sim::dial::<()>(controller_to_dst.clone()).unwrap();

        let client_to_dst = SimAddr::new_with_src(
            AddressProxyPair {
                address: "unix!@client".parse::<ChannelAddr>().unwrap(),
                proxy: "unix!@proxy".parse::<ChannelAddr>().unwrap(),
            },
            "unix!@dst".parse::<ChannelAddr>().unwrap(),
            "unix!@proxy".parse::<ChannelAddr>().unwrap(),
        )
        .unwrap();
        let client_tx = sim::dial::<()>(client_to_dst).unwrap();

        // 1 second of latency
        let simnet_config_yaml = r#"
        edges:
        - src: unix!@controller
          dst: unix!@dst
          metadata:
            latency: 1
        "#;
        update_config(NetworkConfig::from_yaml(simnet_config_yaml).unwrap())
            .await
            .unwrap();

        assert_eq!(SimClock.millis_since_start(RealClock.now()), 0);
        // Fast forward real time to 5 seconds
        tokio::time::advance(tokio::time::Duration::from_secs(5)).await;
        {
            // Send client message
            client_tx.try_post((), oneshot::channel().0).unwrap();
            // Send system message
            controller_tx.try_post((), oneshot::channel().0).unwrap();
            // Allow some time for simnet to run
            RealClock.sleep(tokio::time::Duration::from_secs(1)).await;
        }
        let recs = records().await.unwrap();
        assert_eq!(recs.len(), 2);
        let end_times = recs.iter().map(|rec| rec.end_at).collect::<Vec<_>>();
        // client message was delivered at "real" time = 5 seconds
        assert!(end_times.contains(&5000));
        // system message was delivered at simulated time = 1 second
        assert!(end_times.contains(&1000));
    }
}
