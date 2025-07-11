/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(dead_code)]

//! A simulator capable of simulating Hyperactor's network channels (see: [`channel`]).
//! The simulator can simulate message delivery delays and failures, and is used for
//! testing and development of message distribution techniques.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use async_trait::async_trait;
use dashmap::DashMap;
use dashmap::DashSet;
use enum_as_inner::EnumAsInner;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use serde_with::serde_as;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinError;
use tokio::task::JoinHandle;
use tokio::time::interval;

// for macros
use crate::ActorId;
use crate::Mailbox;
use crate::OncePortRef;
use crate::channel::ChannelAddr;
use crate::clock::Clock;
use crate::clock::RealClock;
use crate::clock::SimClock;
use crate::data::Serialized;

static HANDLE: OnceLock<SimNetHandle> = OnceLock::new();

/// A handle for SimNet through which you can send and schedule events in the
/// network.
///
/// Return the \[`NotStarted`\] error when called before `simnet::start()` has been called
#[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `SimNetError`.
pub fn simnet_handle() -> Result<&'static SimNetHandle, SimNetError> {
    match HANDLE.get() {
        Some(handle) => Ok(handle),
        None => Err(SimNetError::Closed("SimNet not started".to_string())),
    }
}

const OPERATIONAL_MESSAGE_BUFFER_SIZE: usize = 8;

/// This is used to define an Address-type for the network.
/// Addresses are bound to nodes in the network.
pub trait Address: Hash + Debug + Eq + PartialEq + Ord + PartialOrd + Clone {}
impl<A: Hash + Debug + Eq + PartialEq + Ord + PartialOrd + Clone> Address for A {}

type SimulatorTimeInstant = u64;

/// The unit of execution for the simulator.
/// Using handle(), simnet can schedule executions in the network.
/// If you want to send a message for example, you would want to implement
/// a MessageDeliveryEvent much on the lines expressed in simnet tests.
/// You can also do other more advanced concepts such as node churn,
/// or even simulate process spawns in a distributed system. For example,
/// one can implement a SystemActorSimEvent in order to spawn a system
/// actor.
#[async_trait]
pub trait Event: Send + Sync + Debug {
    /// This is the method that will be called when the simulator fires the event
    /// at a particular time instant. Examples:
    /// For messages, it will be delivering the message to the dst's receiver queue.
    /// For a proc spawn, it will be creating the proc object and instantiating it.
    /// For any event that manipulates the network (like adding/removing nodes etc.)
    /// implement handle_network().
    async fn handle(&self) -> Result<(), SimNetError>;

    /// This is the method that will be called when the simulator fires the event
    /// Unless you need to make changes to the network, you do not have to implement this.
    /// Only implement handle() method for all non-simnet requirements.
    async fn handle_network(&self, _phantom: &SimNet) -> Result<(), SimNetError> {
        self.handle().await
    }

    /// The latency of the event. This could be network latency, induced latency (sleep), or
    /// GPU work latency.
    fn duration_ms(&self) -> u64;

    /// Read the simnet config and update self accordingly.
    async fn read_simnet_config(&mut self, _topology: &Arc<Mutex<SimNetConfig>>) {}

    /// A user-friendly summary of the event
    fn summary(&self) -> String;
}

/// This is a simple event that is used to join a node to the network.
/// It is used to bind a node to a channel address.
#[derive(Debug)]
struct NodeJoinEvent {
    channel_addr: ChannelAddr,
}

#[async_trait]
impl Event for NodeJoinEvent {
    async fn handle(&self) -> Result<(), SimNetError> {
        Ok(())
    }

    async fn handle_network(&self, simnet: &SimNet) -> Result<(), SimNetError> {
        simnet.bind(self.channel_addr.clone()).await;
        self.handle().await
    }

    fn duration_ms(&self) -> u64 {
        0
    }

    fn summary(&self) -> String {
        "Node join".into()
    }
}

#[derive(Debug)]
pub(crate) struct SleepEvent {
    done_tx: OncePortRef<()>,
    mailbox: Mailbox,
    duration_ms: u64,
}

impl SleepEvent {
    pub(crate) fn new(done_tx: OncePortRef<()>, mailbox: Mailbox, duration_ms: u64) -> Box<Self> {
        Box::new(Self {
            done_tx,
            mailbox,
            duration_ms,
        })
    }
}

#[async_trait]
impl Event for SleepEvent {
    async fn handle(&self) -> Result<(), SimNetError> {
        Ok(())
    }

    async fn handle_network(&self, _simnet: &SimNet) -> Result<(), SimNetError> {
        self.done_tx
            .clone()
            .send(&self.mailbox, ())
            .map_err(|_err| SimNetError::Closed("TODO".to_string()))?;
        Ok(())
    }

    fn duration_ms(&self) -> u64 {
        self.duration_ms
    }

    fn summary(&self) -> String {
        format!("Sleeping for {} ms", self.duration_ms)
    }
}

#[derive(Debug)]
/// A pytorch operation
pub struct TorchOpEvent {
    op: String,
    done_tx: OncePortRef<()>,
    mailbox: Mailbox,
    args_string: String,
    kwargs_string: String,
    worker_actor_id: ActorId,
}

#[async_trait]
impl Event for TorchOpEvent {
    async fn handle(&self) -> Result<(), SimNetError> {
        Ok(())
    }

    async fn handle_network(&self, _simnet: &SimNet) -> Result<(), SimNetError> {
        self.done_tx
            .clone()
            .send(&self.mailbox, ())
            .map_err(|err| SimNetError::Closed(err.to_string()))?;
        Ok(())
    }

    fn duration_ms(&self) -> u64 {
        100
    }

    fn summary(&self) -> String {
        let kwargs_string = if self.kwargs_string.is_empty() {
            "".to_string()
        } else {
            format!(", {}", self.kwargs_string)
        };
        format!(
            "[{}] Torch Op: {}({}{})",
            self.worker_actor_id, self.op, self.args_string, kwargs_string
        )
    }
}

impl TorchOpEvent {
    /// Creates a new TorchOpEvent.
    pub fn new(
        op: String,
        done_tx: OncePortRef<()>,
        mailbox: Mailbox,
        args_string: String,
        kwargs_string: String,
        worker_actor_id: ActorId,
    ) -> Box<Self> {
        Box::new(Self {
            op,
            done_tx,
            mailbox,
            args_string,
            kwargs_string,
            worker_actor_id,
        })
    }
}

/// Each message is timestamped with the delivery time
/// of the message to the sender.
/// The timestamp is used to determine the order in which
/// messages are delivered to senders.
#[derive(Debug)]
pub(crate) struct ScheduledEvent {
    pub(crate) time: SimulatorTimeInstant,
    pub(crate) event: Box<dyn Event>,
}

/// Dispatcher is a trait that defines the send operation.
/// The send operation takes a target address and a data buffer.
/// This method is called when the simulator is ready for the message to be received
/// by the target address.
#[async_trait]
pub trait Dispatcher<A> {
    /// Send a raw data blob to the given target.
    async fn send(&self, source: Option<A>, target: A, data: Serialized)
    -> Result<(), SimNetError>;
}

#[derive(Hash, Eq, PartialEq, Debug)]
pub(crate) struct SimNetEdge {
    pub(crate) src: ChannelAddr,
    pub(crate) dst: ChannelAddr,
}

#[serde_as]
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SimNetEdgeInfo {
    #[serde_as(as = "serde_with::DurationSeconds<f64>")]
    pub(crate) latency: Duration,
}

/// SimNetError is used to indicate errors that occur during
/// network simulation.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum SimNetError {
    /// An invalid address was encountered.
    #[error("invalid address: {0}")]
    InvalidAddress(String),

    /// An invalid node was encountered.
    #[error("invalid node: {0}")]
    InvalidNode(String, #[source] anyhow::Error),

    /// An invalid parameter was encountered.
    #[error("invalid arg: {0}")]
    InvalidArg(String),

    /// The simulator has been closed.
    #[error("closed: {0}")]
    Closed(String),

    /// Timeout when waiting for something.
    #[error("timeout after {} ms: {}", .0.as_millis(), .1)]
    Timeout(Duration, String),

    /// Cannot deliver the message because destination address is missing.
    #[error("missing destination address")]
    MissingDestinationAddress,

    /// SimnetHandle being accessed without starting simnet
    #[error("simnet not started")]
    NotStarted,
}

struct State {
    // The simnet is allowed to advance to the time of the earliest event in this queue at any time
    scheduled_events: BTreeMap<SimulatorTimeInstant, Vec<ScheduledEvent>>,
    // The simnet is allowed to advance to the time of the earliest event in this queue at any time
    // only if the earliest event in `scheduled_events` occurs after the earliest event in this queue
    // or some debounce period has passed where there are only events in this queue.
    unadvanceable_scheduled_events: BTreeMap<SimulatorTimeInstant, Vec<ScheduledEvent>>,
}

/// The state of the python training script.
#[derive(EnumAsInner, Debug, Serialize, Deserialize, PartialEq, Clone)]
pub enum TrainingScriptState {
    /// The training script is issuing commands
    Running,
    /// The training script is waiting for the backend to return a future result
    Waiting,
}

/// A handle to a running [`SimNet`] instance.
pub struct SimNetHandle {
    join_handle: Mutex<Option<JoinHandle<Vec<SimulatorEventRecord>>>>,
    event_tx: UnboundedSender<(Box<dyn Event>, bool, Option<SimulatorTimeInstant>)>,
    config: Arc<Mutex<SimNetConfig>>,
    pending_event_count: Arc<AtomicUsize>,
    /// A receiver to receive simulator operational messages.
    /// The receiver can be moved out of the simnet handle.
    training_script_state_tx: tokio::sync::watch::Sender<TrainingScriptState>,
    /// Signal to stop the simnet loop
    stop_signal: Arc<AtomicBool>,
}

impl SimNetHandle {
    /// Sends an event to be scheduled onto the simnet's event loop
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `SimNetError`.
    pub fn send_event(&self, event: Box<dyn Event>) -> Result<(), SimNetError> {
        self.send_event_impl(event, true)
    }

    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `SimNetError`.
    fn send_event_impl(&self, event: Box<dyn Event>, advanceable: bool) -> Result<(), SimNetError> {
        self.pending_event_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.event_tx
            .send((event, advanceable, None))
            .map_err(|err| SimNetError::Closed(err.to_string()))
    }

    /// Sends an non-advanceable event to be scheduled onto the simnet's event loop
    /// A non-advanceable event is an event that cannot advance the simnet's time unless
    /// the earliest event in the simnet's advancing event queue occurs after the earliest
    /// event in the simnet's non-advancing event queue, or some debounce period has passed
    /// where there are only events in the simnet's non-advancing event queue.
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `SimNetError`.
    pub fn send_nonadvanceable_event(&self, event: Box<dyn Event>) -> Result<(), SimNetError> {
        self.send_event_impl(event, false)
    }

    /// Sends an event that already has a scheduled time onto the simnet's event loop
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `SimNetError`.
    pub(crate) fn send_scheduled_event(
        &self,
        ScheduledEvent { event, time }: ScheduledEvent,
    ) -> Result<(), SimNetError> {
        self.pending_event_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.event_tx
            .send((event, true, Some(time)))
            .map_err(|err| SimNetError::Closed(err.to_string()))
    }

    /// Let the simnet know if the training script is running or waiting for the backend
    /// to return a future result.
    pub fn set_training_script_state(&self, state: TrainingScriptState) {
        self.training_script_state_tx.send(state).unwrap();
    }

    /// Bind the given address to this simulator instance.
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `SimNetError`.
    pub fn bind(&self, address: ChannelAddr) -> Result<(), SimNetError> {
        self.pending_event_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.event_tx
            .send((
                Box::new(NodeJoinEvent {
                    channel_addr: address,
                }),
                true,
                None,
            ))
            .map_err(|err| SimNetError::Closed(err.to_string()))
    }

    /// Close the simulator, processing pending messages before
    /// completing the returned future.
    pub async fn close(&self) -> Result<Vec<SimulatorEventRecord>, JoinError> {
        // Signal the simnet loop to stop
        self.stop_signal.store(true, Ordering::SeqCst);

        let mut guard = self.join_handle.lock().await;
        if let Some(handle) = guard.take() {
            handle.await
        } else {
            Ok(vec![])
        }
    }

    /// Update the network configuration to SimNet.
    pub async fn update_network_config(&self, config: NetworkConfig) -> Result<(), SimNetError> {
        let guard = &self.config.lock().await.topology;
        for edge in config.edges {
            guard.insert(
                SimNetEdge {
                    src: edge.src.clone(),
                    dst: edge.dst.clone(),
                },
                edge.metadata,
            );
        }
        Ok(())
    }

    /// Wait for all of the received events to be scheduled for flight.
    /// It ticks the simnet time till all of the scheduled events are processed.
    pub async fn flush(&self, timeout: Duration) -> Result<(), SimNetError> {
        let pending_event_count = self.pending_event_count.clone();
        // poll for the pending event count to be zero.
        let mut interval = interval(Duration::from_millis(10));
        let deadline = RealClock.now() + timeout;
        while RealClock.now() < deadline {
            interval.tick().await;
            if pending_event_count.load(std::sync::atomic::Ordering::SeqCst) == 0 {
                return Ok(());
            }
        }
        Err(SimNetError::Timeout(
            timeout,
            "timeout waiting for received events to be scheduled".to_string(),
        ))
    }
}

pub(crate) type Topology = DashMap<SimNetEdge, SimNetEdgeInfo>;

/// Configure network topology for the simnet
pub struct SimNetConfig {
    // For now, we assume the network is fully connected
    // so as to avoid the complexity of maintaining a graph
    // and determining the shortest path between two nodes.
    pub(crate) topology: Topology,
}
/// SimNet defines a network of nodes.
/// Each node is identified by a unique id.
/// The network is represented as a graph of nodes.
/// The graph is represented as a map of edges.
/// The network also has a cloud of inflight messages
pub struct SimNet {
    config: Arc<Mutex<SimNetConfig>>,
    address_book: DashSet<ChannelAddr>,
    state: State,
    max_latency: Duration,
    records: Vec<SimulatorEventRecord>,
    // number of events that has been received but not yet processed.
    pending_event_count: Arc<AtomicUsize>,
}

/// Starts a sim net.
/// Args:
///     max_duration_ms: an optional config to override default settings of the network latency
pub fn start() {
    let max_duration_ms = 1000 * 10;
    // Construct a topology with one node: the default A.
    let address_book: DashSet<ChannelAddr> = DashSet::new();

    let topology = DashMap::new();
    let config = Arc::new(Mutex::new(SimNetConfig { topology }));

    let (training_script_state_tx, training_script_state_rx) =
        tokio::sync::watch::channel(TrainingScriptState::Waiting);
    let (event_tx, event_rx) =
        mpsc::unbounded_channel::<(Box<dyn Event>, bool, Option<SimulatorTimeInstant>)>();
    let pending_event_count = Arc::new(AtomicUsize::new(0));
    let stop_signal = Arc::new(AtomicBool::new(false));

    let join_handle = Mutex::new(Some({
        let config = config.clone();
        let pending_event_count = pending_event_count.clone();
        let stop_signal = stop_signal.clone();

        tokio::spawn(async move {
            SimNet {
                config,
                address_book,
                state: State {
                    scheduled_events: BTreeMap::new(),
                    unadvanceable_scheduled_events: BTreeMap::new(),
                },
                max_latency: Duration::from_millis(max_duration_ms),
                records: Vec::new(),
                pending_event_count,
            }
            .run(event_rx, training_script_state_rx, stop_signal)
            .await
        })
    }));

    HANDLE.get_or_init(|| SimNetHandle {
        join_handle,
        event_tx,
        config,
        pending_event_count,
        training_script_state_tx,
        stop_signal,
    });
}

impl SimNet {
    /// Bind an address to a node id. If node id is not provided, then
    /// randomly choose a node id. If the address is already bound to a node id,
    /// then return the existing node id.
    async fn bind(&self, address: ChannelAddr) {
        // Add if not present.
        if self.address_book.insert(address.clone()) {
            // Add dummy latencies with all the other nodes.
            for other in self.address_book.iter() {
                let duration_ms = if other.key() == &address {
                    1
                } else {
                    rand::random::<u64>() % self.max_latency.as_millis() as u64 + 1
                };
                let latency = Duration::from_millis(duration_ms);
                let guard = &self.config.lock().await.topology;
                guard.insert(
                    SimNetEdge {
                        src: address.clone(),
                        dst: other.clone(),
                    },
                    SimNetEdgeInfo { latency },
                );
                if address != *other.key() {
                    guard.insert(
                        SimNetEdge {
                            src: other.clone(),
                            dst: address.clone(),
                        },
                        SimNetEdgeInfo { latency },
                    );
                }
            }
        }
    }

    async fn create_scheduled_event(&mut self, mut event: Box<dyn Event>) -> ScheduledEvent {
        // Get latency
        event.read_simnet_config(&self.config).await;
        ScheduledEvent {
            time: SimClock.millis_since_start(
                SimClock.now() + tokio::time::Duration::from_millis(event.duration_ms()),
            ),
            event,
        }
    }

    /// Schedule the event into the network.
    fn schedule_event(&mut self, scheduled_event: ScheduledEvent, advanceable: bool) {
        let start_at = SimClock.millis_since_start(SimClock.now());
        let end_at = scheduled_event.time;

        self.records.push(SimulatorEventRecord {
            summary: scheduled_event.event.summary(),
            start_at,
            end_at,
        });

        if advanceable {
            self.state
                .scheduled_events
                .entry(scheduled_event.time)
                .or_insert_with(Vec::new)
                .push(scheduled_event);
        } else {
            self.state
                .unadvanceable_scheduled_events
                .entry(scheduled_event.time)
                .or_insert_with(Vec::new)
                .push(scheduled_event);
        }
    }

    /// Run the simulation. This will dispatch all the messages in the network.
    /// And wait for new ones.
    async fn run(
        &mut self,
        mut event_rx: UnboundedReceiver<(Box<dyn Event>, bool, Option<SimulatorTimeInstant>)>,
        training_script_state_rx: tokio::sync::watch::Receiver<TrainingScriptState>,
        stop_signal: Arc<AtomicBool>,
    ) -> Vec<SimulatorEventRecord> {
        // The simulated number of milliseconds the training script
        // has spent waiting for the backend to resolve a future
        let mut training_script_waiting_time: u64 = 0;
        // Duration elapsed while only non_advanceable_events has events
        let mut debounce_timer: Option<tokio::time::Instant> = None;
        'outer: loop {
            // Check if we should stop
            if stop_signal.load(Ordering::SeqCst) {
                break 'outer self.records.clone();
            }

            while let Ok(Some((event, advanceable, time))) = RealClock
                .timeout(tokio::time::Duration::from_millis(1), event_rx.recv())
                .await
            {
                let scheduled_event = match time {
                    Some(time) => ScheduledEvent {
                        time: time + training_script_waiting_time,
                        event,
                    },
                    None => self.create_scheduled_event(event).await,
                };
                self.schedule_event(scheduled_event, advanceable);
            }

            {
                // If the training script is runnning and issuing commands
                // it is not safe to advance past the training script time
                // otherwise a command issued by the training script may
                // be scheduled for a time in the past
                if training_script_state_rx.borrow().is_running()
                    && self
                        .state
                        .scheduled_events
                        .first_key_value()
                        .is_some_and(|(time, _)| {
                            *time
                                > SimClock.millis_since_start(RealClock.now())
                                    + training_script_waiting_time
                        })
                {
                    tokio::task::yield_now().await;
                    continue;
                }
                match (
                    self.state.scheduled_events.first_key_value(),
                    self.state.unadvanceable_scheduled_events.first_key_value(),
                ) {
                    (None, Some(_)) if debounce_timer.is_none() => {
                        // Start debounce timer when only the non-advancedable
                        // queue has events and the timer has not already started
                        debounce_timer = Some(RealClock.now());
                    }
                    // Timer already active
                    (None, Some(_)) => {}
                    // Reset timer when non-advanceable queue is not the only queue with events
                    _ => {
                        debounce_timer = None;
                    }
                }
                // process for next delivery time.
                let Some((scheduled_time, scheduled_events)) = (match (
                    self.state.scheduled_events.first_key_value(),
                    self.state.unadvanceable_scheduled_events.first_key_value(),
                ) {
                    (Some((advanceable_time, _)), Some((unadvanceable_time, _))) => {
                        if unadvanceable_time < advanceable_time {
                            self.state.unadvanceable_scheduled_events.pop_first()
                        } else {
                            self.state.scheduled_events.pop_first()
                        }
                    }
                    (Some(_), None) => self.state.scheduled_events.pop_first(),
                    (None, Some(_)) => match debounce_timer {
                        Some(time) => {
                            if time.elapsed() > tokio::time::Duration::from_millis(1000) {
                                // debounce interval has elapsed, reset timer
                                debounce_timer = None;
                                self.state.unadvanceable_scheduled_events.pop_first()
                            } else {
                                None
                            }
                        }
                        None => None,
                    },
                    (None, None) => None,
                }) else {
                    tokio::select! {
                        Some((event, advanceable, time)) = event_rx.recv() => {
                            let scheduled_event = match time {
                                Some(time) => ScheduledEvent {
                                    time: time + training_script_waiting_time,
                                    event,
                                },
                                None => self.create_scheduled_event(event).await,
                            };
                            self.schedule_event(scheduled_event, advanceable);
                        },
                        _ = RealClock.sleep(Duration::from_millis(10)) => {}
                    }
                    continue;
                };
                if training_script_state_rx.borrow().is_waiting() {
                    let advanced_time =
                        scheduled_time - SimClock.millis_since_start(SimClock.now());
                    training_script_waiting_time += advanced_time;
                }
                SimClock.advance_to(scheduled_time);
                for scheduled_event in scheduled_events {
                    self.pending_event_count
                        .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    if scheduled_event.event.handle_network(self).await.is_err() {
                        break 'outer self.records.clone(); //TODO
                    }
                }
            }
        }
    }
}

fn serialize_optional_channel_addr<S>(
    addr: &Option<ChannelAddr>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match addr {
        Some(addr) => serializer.serialize_str(&addr.to_string()),
        None => serializer.serialize_none(),
    }
}

fn deserialize_channel_addr<'de, D>(deserializer: D) -> Result<ChannelAddr, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

/// DeliveryRecord is a structure to bookkeep the message events.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SimulatorEventRecord {
    /// Event dependent summary for user
    pub summary: String,
    /// The time at which the message delivery was started.
    pub start_at: SimulatorTimeInstant,
    /// The time at which the message was delivered to the receiver.
    pub end_at: SimulatorTimeInstant,
}

/// A configuration for the network topology.
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    edges: Vec<EdgeConfig>,
}

/// A configuration for the network edge.
#[derive(Debug, Serialize, Deserialize)]
pub struct EdgeConfig {
    #[serde(deserialize_with = "deserialize_channel_addr")]
    src: ChannelAddr,
    #[serde(deserialize_with = "deserialize_channel_addr")]
    dst: ChannelAddr,
    metadata: SimNetEdgeInfo,
}

impl NetworkConfig {
    /// Create a new configuration from a YAML string.
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `SimNetError`.
    pub fn from_yaml(yaml: &str) -> Result<Self, SimNetError> {
        let config: NetworkConfig = serde_yaml::from_str(yaml)
            .map_err(|err| SimNetError::InvalidArg(format!("failed to parse config: {}", err)))?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use async_trait::async_trait;
    use tokio::sync::Mutex;

    use super::*;
    use crate::channel::sim::SimAddr;
    use crate::clock::Clock;
    use crate::clock::RealClock;
    use crate::clock::SimClock;
    use crate::data::Serialized;
    use crate::id;
    use crate::simnet;
    use crate::simnet::Dispatcher;
    use crate::simnet::Event;
    use crate::simnet::SimNetError;

    #[derive(Debug)]
    struct MessageDeliveryEvent {
        src_addr: SimAddr,
        dest_addr: SimAddr,
        data: Serialized,
        duration_ms: u64,
        dispatcher: Option<TestDispatcher>,
    }

    #[async_trait]
    impl Event for MessageDeliveryEvent {
        async fn handle(&self) -> Result<(), simnet::SimNetError> {
            if let Some(dispatcher) = &self.dispatcher {
                dispatcher
                    .send(
                        Some(self.src_addr.clone()),
                        self.dest_addr.clone(),
                        self.data.clone(),
                    )
                    .await?;
            }
            Ok(())
        }
        fn duration_ms(&self) -> u64 {
            self.duration_ms
        }

        fn summary(&self) -> String {
            format!(
                "Sending message from {} to {}",
                self.src_addr.addr().clone(),
                self.dest_addr.addr().clone()
            )
        }

        async fn read_simnet_config(&mut self, config: &Arc<Mutex<SimNetConfig>>) {
            let edge = SimNetEdge {
                src: self.src_addr.addr().clone(),
                dst: self.dest_addr.addr().clone(),
            };
            self.duration_ms = config
                .lock()
                .await
                .topology
                .get(&edge)
                .map_or_else(|| 1, |v| v.latency.as_millis() as u64);
        }
    }

    impl MessageDeliveryEvent {
        fn new(
            src_addr: SimAddr,
            dest_addr: SimAddr,
            data: Serialized,
            dispatcher: Option<TestDispatcher>,
        ) -> Self {
            Self {
                src_addr,
                dest_addr,
                data,
                duration_ms: 1,
                dispatcher,
            }
        }
    }

    #[derive(Debug, Clone)]
    struct TestDispatcher {
        pub mbuffers: Arc<Mutex<HashMap<SimAddr, Vec<Serialized>>>>,
    }

    impl Default for TestDispatcher {
        fn default() -> Self {
            Self {
                mbuffers: Arc::new(Mutex::new(HashMap::new())),
            }
        }
    }

    #[async_trait]
    impl Dispatcher<SimAddr> for TestDispatcher {
        async fn send(
            &self,
            _source: Option<SimAddr>,
            target: SimAddr,
            data: Serialized,
        ) -> Result<(), SimNetError> {
            let mut buf = self.mbuffers.lock().await;
            buf.entry(target).or_default().push(data);
            Ok(())
        }
    }

    #[cfg(target_os = "linux")]
    fn random_abstract_addr() -> ChannelAddr {
        use rand::Rng;
        use rand::distributions::Alphanumeric;

        let random_string = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(24)
            .map(char::from)
            .collect::<String>();
        format!("unix!@{random_string}").parse().unwrap()
    }

    #[tokio::test]
    async fn test_handle_instantiation() {
        start();
        simnet_handle().unwrap().close().await.unwrap();
    }

    #[tokio::test]
    async fn test_simnet_config() {
        // Tests that we can create a simnet, config latency between two node and deliver
        // the message with configured latency.
        start();
        let alice = "local!1".parse::<simnet::ChannelAddr>().unwrap();
        let bob = "local!2".parse::<simnet::ChannelAddr>().unwrap();
        let latency = Duration::from_millis(1000);
        let config = NetworkConfig {
            edges: vec![EdgeConfig {
                src: alice.clone(),
                dst: bob.clone(),
                metadata: SimNetEdgeInfo { latency },
            }],
        };
        simnet_handle()
            .unwrap()
            .update_network_config(config)
            .await
            .unwrap();

        let alice = SimAddr::new(alice).unwrap();
        let bob = SimAddr::new(bob).unwrap();
        let msg = Box::new(MessageDeliveryEvent::new(
            alice,
            bob,
            Serialized::serialize(&"123".to_string()).unwrap(),
            None,
        ));
        simnet_handle().unwrap().send_event(msg).unwrap();
        simnet_handle()
            .unwrap()
            .flush(Duration::from_secs(30))
            .await
            .unwrap();
        let records = simnet_handle().unwrap().close().await;
        let expected_record = SimulatorEventRecord {
            summary: "Sending message from local!1 to local!2".to_string(),
            start_at: 0,
            end_at: latency.as_millis() as u64,
        };
        assert!(records.as_ref().unwrap().len() == 1);
        assert_eq!(records.unwrap().first().unwrap(), &expected_record);
    }

    #[tokio::test]
    async fn test_simnet_debounce() {
        start();
        let alice = "local!1".parse::<simnet::ChannelAddr>().unwrap();
        let bob = "local!2".parse::<simnet::ChannelAddr>().unwrap();

        let latency = Duration::from_millis(10000);
        simnet_handle()
            .unwrap()
            .update_network_config(NetworkConfig {
                edges: vec![EdgeConfig {
                    src: alice.clone(),
                    dst: bob.clone(),
                    metadata: SimNetEdgeInfo { latency },
                }],
            })
            .await
            .unwrap();

        let alice = SimAddr::new(alice).unwrap();
        let bob = SimAddr::new(bob).unwrap();

        // Rapidly send 10 messages expecting that each one debounces the processing
        for _ in 0..10 {
            simnet_handle()
                .unwrap()
                .send_event(Box::new(MessageDeliveryEvent::new(
                    alice.clone(),
                    bob.clone(),
                    Serialized::serialize(&"123".to_string()).unwrap(),
                    None,
                )))
                .unwrap();
            RealClock
                .sleep(tokio::time::Duration::from_micros(500))
                .await;
        }

        simnet_handle()
            .unwrap()
            .flush(Duration::from_secs(20))
            .await
            .unwrap();

        let records = simnet_handle().unwrap().close().await;
        assert_eq!(records.as_ref().unwrap().len(), 10);

        // If debounce is successful, the simnet will not advance to the delivery of any of
        // the messages before all are received
        assert_eq!(
            records.unwrap().last().unwrap().end_at,
            latency.as_millis() as u64
        );
    }

    #[tokio::test]
    async fn test_sim_dispatch() {
        start();
        let sender = Some(TestDispatcher::default());
        let mut addresses: Vec<simnet::ChannelAddr> = Vec::new();
        // // Create a simple network of 4 nodes.
        for i in 0..4 {
            addresses.push(
                format!("local!{}", i)
                    .parse::<simnet::ChannelAddr>()
                    .unwrap(),
            );
        }

        let messages: Vec<Serialized> = vec!["First 0 1", "First 2 3", "Second 0 1"]
            .into_iter()
            .map(|s| Serialized::serialize(&s.to_string()).unwrap())
            .collect();

        let addr_0 = SimAddr::new(addresses[0].clone()).unwrap();
        let addr_1 = SimAddr::new(addresses[1].clone()).unwrap();
        let addr_2 = SimAddr::new(addresses[2].clone()).unwrap();
        let addr_3 = SimAddr::new(addresses[3].clone()).unwrap();
        let one = Box::new(MessageDeliveryEvent::new(
            addr_0.clone(),
            addr_1.clone(),
            messages[0].clone(),
            sender.clone(),
        ));
        let two = Box::new(MessageDeliveryEvent::new(
            addr_2.clone(),
            addr_3.clone(),
            messages[1].clone(),
            sender.clone(),
        ));
        let three = Box::new(MessageDeliveryEvent::new(
            addr_0.clone(),
            addr_1.clone(),
            messages[2].clone(),
            sender.clone(),
        ));

        simnet_handle().unwrap().send_event(one).unwrap();
        simnet_handle().unwrap().send_event(two).unwrap();
        simnet_handle().unwrap().send_event(three).unwrap();

        simnet_handle()
            .unwrap()
            .flush(Duration::from_millis(1000))
            .await
            .unwrap();
        let records = simnet_handle().unwrap().close().await.unwrap();
        eprintln!("Records: {:?}", records);
        // Close the channel
        simnet_handle().unwrap().close().await.unwrap();

        // Check results
        let buf = sender.as_ref().unwrap().mbuffers.lock().await;
        assert_eq!(buf.len(), 2);
        assert_eq!(buf[&addr_1].len(), 2);
        assert_eq!(buf[&addr_3].len(), 1);

        assert_eq!(buf[&addr_1][0], messages[0]);
        assert_eq!(buf[&addr_1][1], messages[2]);
        assert_eq!(buf[&addr_3][0], messages[1]);
    }

    #[tokio::test]
    async fn test_read_config_from_yaml() {
        let yaml = r#"
 edges:
   - src: local!0
     dst: local!1
     metadata:
       latency: 1
   - src: local!0
     dst: local!2
     metadata:
       latency: 2
   - src: local!1
     dst: local!2
     metadata:
       latency: 3
 "#;
        let config = NetworkConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.edges.len(), 3);
        assert_eq!(
            config.edges[0].src,
            "local!0".parse::<simnet::ChannelAddr>().unwrap()
        );
        assert_eq!(
            config.edges[0].dst,
            "local!1".parse::<simnet::ChannelAddr>().unwrap()
        );
        assert_eq!(config.edges[0].metadata.latency, Duration::from_secs(1));
        assert_eq!(
            config.edges[1].src,
            "local!0".parse::<simnet::ChannelAddr>().unwrap()
        );
        assert_eq!(
            config.edges[1].dst,
            "local!2".parse::<simnet::ChannelAddr>().unwrap()
        );
        assert_eq!(config.edges[1].metadata.latency, Duration::from_secs(2));
        assert_eq!(
            config.edges[2].src,
            "local!1".parse::<simnet::ChannelAddr>().unwrap()
        );
        assert_eq!(
            config.edges[2].dst,
            "local!2".parse::<simnet::ChannelAddr>().unwrap()
        );
        assert_eq!(config.edges[2].metadata.latency, Duration::from_secs(3));
    }

    #[tokio::test]
    async fn test_sim_sleep() {
        start();

        let start = SimClock.now();
        assert_eq!(SimClock.millis_since_start(start), 0);

        SimClock.sleep(tokio::time::Duration::from_secs(10)).await;

        let end = SimClock.now();
        assert_eq!(SimClock.millis_since_start(end), 10000);
    }

    #[tokio::test]
    async fn test_torch_op() {
        start();
        let args_string = "1, 2".to_string();
        let kwargs_string = "a=2".to_string();

        let mailbox = Mailbox::new_detached(id!(proc[0].proc).clone());
        let (tx, rx) = mailbox.open_once_port::<()>();

        simnet_handle()
            .unwrap()
            .send_event(TorchOpEvent::new(
                "torch.ops.aten.ones.default".to_string(),
                tx.bind(),
                mailbox,
                args_string,
                kwargs_string,
                id!(mesh_0_worker[0].worker_0),
            ))
            .unwrap();

        rx.recv().await.unwrap();

        simnet_handle()
            .unwrap()
            .flush(Duration::from_millis(1000))
            .await
            .unwrap();
        let records = simnet_handle().unwrap().close().await;
        let expected_record = SimulatorEventRecord {
            summary:
                "[mesh_0_worker[0].worker_0[0]] Torch Op: torch.ops.aten.ones.default(1, 2, a=2)"
                    .to_string(),
            start_at: 0,
            end_at: 100,
        };
        assert!(records.as_ref().unwrap().len() == 1);
        assert_eq!(records.unwrap().first().unwrap(), &expected_record);
    }
}
