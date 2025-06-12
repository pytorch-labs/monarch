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
use rand::RngCore;
use regex::Regex;
use tokio::sync::Mutex;

use super::*;
use crate::ActorId;
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
use crate::simnet::PerfettoFlow;
use crate::simnet::PerfettoTrace;
use crate::simnet::SimNetConfig;
use crate::simnet::SimNetEdge;
use crate::simnet::SimNetError;
use crate::simnet::simnet_handle;

lazy_static! {
    static ref SENDER: SimDispatcher = SimDispatcher::default();
}
static SIM_LINK_BUF_SIZE: usize = 256;

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
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `SimNetError`.
    pub fn new_with_src(
        src: AddressProxyPair,
        addr: ChannelAddr,
        proxy: ChannelAddr,
    ) -> Result<Self, SimNetError> {
        Self::new_impl(Some(Box::new(src)), addr, proxy)
    }

    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `SimNetError`.
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

fn parse_message(input: &str) -> String {
    let open_brace_pos = match input.find('{') {
        Some(pos) => pos,
        None => return input.to_string(),
    };

    let first_part = input[..open_brace_pos].to_string();

    let start_quote_pos = match input[open_brace_pos + 1..].find('"') {
        Some(pos) => pos + open_brace_pos + 1,
        None => return input.to_string(),
    };

    let end_quote_pos = match input[start_quote_pos + 1..].find('"') {
        Some(pos) => pos + start_quote_pos + 1,
        None => return input.to_string(),
    };

    let second_part = input[start_quote_pos + 1..end_quote_pos].to_string();

    format!("{}::{}", first_part, second_part)
}

/// Message Event that can be passed around in the simnet.
#[derive(Debug)]
pub(crate) struct MessageSendEvent {
    src_addr: Option<AddressProxyPair>,
    dest_addr: AddressProxyPair,
    data: Serialized,
    duration_ms: u64,
    inflight_time_ms: u64,
    // Used to match a pair of send/recv events.
    id: String,
    sender_actor: ActorId,
    dest_actor: ActorId,
    message_type: String,
}

impl MessageSendEvent {
    /// Creates a new MessageSendEvent.
    pub fn new(
        src_addr: Option<AddressProxyPair>,
        dest_addr: AddressProxyPair,
        data: Serialized,
    ) -> Self {
        let (sender_actor, dest_actor, message_type) =
            if let Ok(envelope) = data.deserialized::<MessageEnvelope>() {
                let msg_string = envelope.data().to_string();
                let parsed = parse_message(&msg_string);
                (
                    envelope.sender().clone(),
                    envelope.dest().actor_id().clone(),
                    parsed,
                )
            } else {
                (
                    id!(unknown[0].unknown),
                    id!(unknown[0].unknown),
                    "unknown".to_string(),
                )
            };

        Self {
            src_addr,
            dest_addr,
            data,
            duration_ms: 1,
            inflight_time_ms: 1,
            id: format!("0x{}", random_hex_str()),
            sender_actor,
            dest_actor,
            message_type,
        }
    }
}

#[async_trait]
impl Event for MessageSendEvent {
    async fn handle(&self) -> Result<(), SimNetError> {
        let inflight_time_ms = self.inflight_time_ms;
        let event = Box::new(MessageRecvEvent::new(
            self.src_addr.clone(),
            self.dest_addr.clone(),
            self.data.clone(),
            Some(self.id.clone()),
        ));

        tokio::task::spawn(async move {
            SimClock
                .sleep(tokio::time::Duration::from_millis(inflight_time_ms))
                .await;

            if let Ok(handle) = simnet_handle() {
                let _ = handle.send_event(event);
            }
        });

        Ok(())
    }

    fn duration_ms(&self) -> u64 {
        self.duration_ms
    }

    async fn read_simnet_config(&mut self, topology: &Arc<Mutex<SimNetConfig>>) {
        if let Some(src_addr) = &self.src_addr {
            let edge = SimNetEdge {
                src: src_addr.address.clone(),
                dst: self.dest_addr.address.clone(),
            };
            self.inflight_time_ms = topology
                .lock()
                .await
                .topology
                .get(&edge)
                .map_or_else(|| 1, |v| v.latency.as_millis() as u64);
        }
    }

    fn to_perfetto(&self, start: u64, _end: u64) -> Option<PerfettoTrace> {
        Some(PerfettoTrace {
            name: format!("send {}", self.message_type),
            cat: "message".to_string(),
            ph: "X".to_string(),
            ts: start * 1000,
            dur: 1000,
            actor_id: self.sender_actor.clone(),
            bind_id: Some(self.id.clone()),
            flow: Some(PerfettoFlow::Out),
        })
    }
}

/// Message Recv Event that can be passed around in the simnet.
#[derive(Debug)]
pub(crate) struct MessageRecvEvent {
    src_addr: Option<AddressProxyPair>,
    dest_addr: AddressProxyPair,
    data: Serialized,
    duration_ms: u64,
    // Used to match a pair of send/recv events.
    id: Option<String>,
    sender_actor: ActorId,
    dest_actor: ActorId,
    message_type: String,
}

impl MessageRecvEvent {
    /// Creates a new MessageRecvEvent.
    pub fn new(
        src_addr: Option<AddressProxyPair>,
        dest_addr: AddressProxyPair,
        data: Serialized,
        id: Option<String>,
    ) -> Self {
        let (sender_actor, dest_actor, message_type) =
            if let Ok(envelope) = data.deserialized::<MessageEnvelope>() {
                let msg_string = envelope.data().to_string();
                let parsed = parse_message(&msg_string);
                (
                    envelope.sender().clone(),
                    envelope.dest().actor_id().clone(),
                    parsed,
                )
            } else {
                (
                    hyperactor::id!(unknown[0].unknown),
                    hyperactor::id!(unknown[0].unknown),
                    "unknown".to_string(),
                )
            };

        Self {
            src_addr,
            dest_addr,
            data,
            duration_ms: 1,
            id,
            sender_actor,
            dest_actor,
            message_type,
        }
    }
}

#[async_trait]
impl Event for MessageRecvEvent {
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

    fn to_perfetto(&self, start: u64, _end: u64) -> Option<PerfettoTrace> {
        Some(PerfettoTrace {
            name: format!("recv {}", self.message_type),
            cat: "message".to_string(),
            ph: "X".to_string(),
            ts: start * 1000,
            dur: 1000,
            actor_id: self.dest_actor.clone(),
            bind_id: self.id.clone(),
            flow: Some(PerfettoFlow::In),
        })
    }
}

fn random_hex_str() -> String {
    let mut bytes = vec![0u8; 10 / 2];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes
        .into_iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

/// Bind a channel address to the simnet. It will register the address as a node in simnet,
/// and configure default latencies between this node and all other existing nodes.
pub async fn bind(addr: ChannelAddr) -> anyhow::Result<(), SimNetError> {
    simnet_handle()?.bind(addr)
}

/// Update the configuration for simnet.
pub async fn update_config(config: simnet::NetworkConfig) -> anyhow::Result<(), SimNetError> {
    // Only update network config for now, will add host config in the future.
    simnet_handle()?.update_network_config(config).await
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
#[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `SimNetError`.
fn is_external_addr(addr: &AddressProxyPair) -> anyhow::Result<bool, SimNetError> {
    Ok(simnet_handle()?.proxy_addr() != &addr.proxy)
}

#[async_trait]
impl Dispatcher<AddressProxyPair> for SimDispatcher {
    async fn send(
        &self,
        _src_addr: Option<AddressProxyPair>,
        addr: AddressProxyPair,
        data: Serialized,
    ) -> Result<(), SimNetError> {
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
        match simnet_handle() {
            Ok(handle) => match &self.src_addr {
                Some(src_addr) if src_addr.proxy != *handle.proxy_addr() => {
                    let event = Box::new(MessageRecvEvent::new(
                        self.src_addr.clone(),
                        self.dst_addr.clone(),
                        data,
                        None,
                    ));
                    let recv_time = RealClock.now();

                    tokio::task::spawn(async move {
                        SimClock.sleep_until(recv_time).await;

                        if let Ok(handle) = simnet_handle() {
                            let _ = handle.send_event(event);
                        }
                    });
                    Ok(())
                }
                _ => handle.send_event(Box::new(MessageSendEvent::new(
                    self.src_addr.clone(),
                    self.dst_addr.clone(),
                    data,
                ))),
            }
            .map_err(|err: SimNetError| SendError(ChannelError::from(err), message)),
            Err(err) => Err(SendError(ChannelError::from(err), message)),
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
    use crate::PortId;
    use crate::clock::Clock;
    use crate::clock::RealClock;
    use crate::clock::SimClock;
    use crate::id;
    use crate::simnet::NetworkConfig;
    use crate::simnet::ProxyMessage;
    use crate::simnet::start;

    #[tokio::test]
    async fn test_sim_basic() {
        let dst_ok = vec!["[::1]:1234", "tcp!127.0.0.1:8080", "local!123"];
        let srcs_ok = vec!["[::2]:1234", "tcp!127.0.0.2:8080", "local!124"];

        let proxy = ChannelAddr::any(ChannelTransport::Unix);
        start(
            ChannelAddr::any(ChannelTransport::Unix),
            proxy.clone(),
            1000,
        )
        .unwrap();

        // TODO: New NodeAdd event should do this for you..
        for addr in dst_ok.iter().chain(srcs_ok.iter()) {
            // Add to network along with its edges.
            simnet_handle()
                .unwrap()
                .bind(addr.parse::<ChannelAddr>().unwrap())
                .unwrap();
        }
        // Messages are transferred internally if only there's a local proxy and the
        // dst proxy is the same as local proxy.
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

        let records = sim::simnet_handle().unwrap().close().await.unwrap();
        eprintln!("records: {:#?}", records);
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
        let proxy: ChannelAddr = "unix!@proxy".parse().unwrap();
        start(
            ChannelAddr::any(ChannelTransport::Unix),
            proxy.clone(),
            1000,
        )
        .unwrap();

        tokio::time::pause();
        let sim_addr =
            SimAddr::new("unix!@dst".parse::<ChannelAddr>().unwrap(), proxy.clone()).unwrap();
        let sim_addr_with_src = SimAddr::new_with_src(
            AddressProxyPair {
                address: "unix!@src".parse::<ChannelAddr>().unwrap(),
                proxy: proxy.clone(),
            },
            "unix!@dst".parse::<ChannelAddr>().unwrap(),
            proxy.clone(),
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
            // Allow simnet to run
            tokio::task::yield_now().await;
            // Messages have not been receive since 10 seconds have not elapsed
            assert!(rx.rx.try_recv().is_err());
        }
        tokio::time::advance(
            // Advance "real" time by 1 ms for send time
            tokio::time::Duration::from_millis(1)
            // Advance "real" time by 100 seconds for inflight time
            + tokio::time::Duration::from_secs(100)
            // Advance "real" time by 1 ms for recv time
            + tokio::time::Duration::from_millis(1),
        )
        .await;
        {
            // Allow some time for simnet to run
            tokio::task::yield_now().await;
            // Messages are received
            assert!(rx.rx.try_recv().is_ok());
        }
    }

    #[tokio::test]
    async fn test_client_message_scheduled_realtime() {
        tokio::time::pause();
        let proxy_addr = ChannelAddr::any(ChannelTransport::Unix);
        start(
            ChannelAddr::any(ChannelTransport::Unix),
            proxy_addr.clone(),
            1000,
        )
        .unwrap();

        let client_to_dst = SimAddr::new_with_src(
            AddressProxyPair {
                address: ChannelAddr::any(ChannelTransport::Unix),
                proxy: ChannelAddr::any(ChannelTransport::Unix),
            },
            "unix!@dst".parse::<ChannelAddr>().unwrap(),
            proxy_addr.clone(),
        )
        .unwrap();
        let client_tx = sim::dial::<()>(client_to_dst).unwrap();

        assert_eq!(SimClock.millis_since_start(RealClock.now()), 0);
        // Fast forward real time to 5 seconds
        tokio::time::advance(tokio::time::Duration::from_secs(5)).await;
        {
            // Send client message
            client_tx.try_post((), oneshot::channel().0).unwrap();
            tokio::time::advance(tokio::time::Duration::from_millis(1)).await;
            // Allow some time for simnet to run
            tokio::task::yield_now().await;
        }
        let recs = simnet::simnet_handle().unwrap().close().await.unwrap();
        let recs = recs
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["ph"] != "M")
            .collect::<Vec<_>>();
        // client message was delivered at "real" time = 5 seconds
        assert_eq!(
            recs.first().map(|rec| rec["ts"].as_u64().unwrap()).unwrap(),
            5000 * 1000
        );
    }
}
