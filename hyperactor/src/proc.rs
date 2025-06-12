/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! This module provides [`Proc`], which is the runtime used within a single
//! proc.

// TODO: define a set of proc errors and plumb these throughout

use std::any::Any;
use std::any::TypeId;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::hash::Hash;
use std::hash::Hasher;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::Weak;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::SystemTime;

use async_trait::async_trait;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use dashmap::mapref::multiple::RefMulti;
use futures::FutureExt;
use hyperactor_telemetry::recorder;
use hyperactor_telemetry::recorder::Recording;
use serde::Deserialize;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::Instrument;

use crate as hyperactor;
use crate::Actor;
use crate::ActorRef;
use crate::Handler;
use crate::Message;
use crate::Named;
use crate::RemoteMessage;
use crate::actor::ActorError;
use crate::actor::ActorErrorKind;
use crate::actor::ActorHandle;
use crate::actor::ActorStatus;
use crate::actor::Binds;
use crate::actor::RemoteActor;
use crate::actor::RemoteHandles;
use crate::actor::Signal;
use crate::cap;
use crate::clock::Clock;
use crate::clock::ClockKind;
use crate::data::Serialized;
use crate::data::TypeInfo;
use crate::mailbox::BoxedMailboxSender;
use crate::mailbox::DeliveryError;
use crate::mailbox::Mailbox;
use crate::mailbox::MailboxMuxer;
use crate::mailbox::MailboxSender;
use crate::mailbox::MessageEnvelope;
use crate::mailbox::OncePortHandle;
use crate::mailbox::OncePortReceiver;
use crate::mailbox::PanickingMailboxSender;
use crate::mailbox::PortHandle;
use crate::mailbox::PortReceiver;
use crate::mailbox::Undeliverable;
use crate::metrics::MESSAGE_HANDLER_DURATION;
use crate::metrics::MESSAGE_QUEUE_SIZE;
use crate::panic_handler;
use crate::reference::ActorId;
use crate::reference::Index;
use crate::reference::PortId;
use crate::reference::ProcId;
use crate::reference::id;
use crate::supervision::ActorSupervisionEvent;

/// This is used to mint new local ranks for [`Proc::local`].
static NEXT_LOCAL_RANK: AtomicUsize = AtomicUsize::new(0);

/// A proc instance is the runtime managing a single proc in Hyperactor.
/// It is responsible for spawning actors in the proc, multiplexing messages
/// to/within actors in the proc, and providing fallback routing to external
/// procs.
///
/// Procs are also responsible for maintaining the local supervision hierarchy.
#[derive(Clone, Debug)]
pub struct Proc(Arc<ProcState>);

#[derive(Debug)]
struct ProcState {
    /// The proc's id. This should be globally unique, but is not (yet)
    /// for local-only procs.
    proc_id: ProcId,

    /// A muxer instance that has entries for every actor managed by
    /// the proc.
    proc_muxer: MailboxMuxer,

    /// Sender used to forward messages outside of the proc.
    forwarder: BoxedMailboxSender,

    /// All of the roots (i.e., named actors with pid=0) in the proc.
    /// These are also known as "global actors", since they may be
    /// spawned remotely.
    roots: DashMap<String, AtomicUsize>,

    /// Keep track of all of the active actors in the proc.
    ledger: ActorLedger,

    /// Used by root actors to send events to the actor coordinating
    /// supervision of root actors in this proc.
    supervision_coordinator_port: OnceLock<PortHandle<ActorSupervisionEvent>>,

    clock: ClockKind,
}

/// A snapshot view of the proc's actor ledger.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActorLedgerSnapshot {
    /// All the actor trees in the proc, mapping the root id to the root
    /// of each tree.
    pub roots: HashMap<ActorId, ActorTreeSnapshot>,
}

/// A event for one row of log.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Event {
    /// Time when the event happend.
    pub time: SystemTime,
    /// The payload of the event.
    pub fields: Vec<(String, recorder::Value)>,
    /// The sequence number of the event.
    pub seq: usize,
}

impl From<recorder::Event> for Event {
    fn from(event: recorder::Event) -> Event {
        Event {
            time: event.time,
            fields: event.fields(),
            seq: event.seq,
        }
    }
}

/// A snapshot of an actor tree (rooted at a pid=0 actor).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActorTreeSnapshot {
    /// The PID of this actor.
    pub pid: Index,

    /// The type name of the actor. If the actor is [`crate::Named`], then
    /// this is the registered name; otherwise it is the actor type's
    /// [`std::any::type_name`].
    pub type_name: String,

    /// The actor's current status.
    pub status: ActorStatus,

    /// Various operational stats for the actor.
    pub stats: ActorStats,

    /// This actor's handlers, mapping port numbers to the named type handled.
    pub handlers: HashMap<u64, String>,

    /// This actor's children.
    pub children: HashMap<Index, ActorTreeSnapshot>,

    /// Recent events emitted by the actor's logging.
    pub events: Vec<Event>,

    /// The current set of spans entered by the actor. These should be active
    /// only while the actor is entered in a handler.
    pub spans: Vec<Vec<String>>,
}

impl Hash for ActorTreeSnapshot {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pid.hash(state);
    }
}

/// Operational stats for an actor instance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[derive(Default)]
pub struct ActorStats {
    /// The number of messages processed by the actor.
    num_processed_messages: u64,
}

impl fmt::Display for ActorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "num_processed_messages={}", self.num_processed_messages)
    }
}

#[derive(Debug)]
struct ActorLedger {
    // Root actors. Map's value is its key's InstanceCell.
    roots: DashMap<ActorId, WeakInstanceCell>,
}

impl ActorLedger {
    fn new() -> Self {
        Self {
            roots: DashMap::new(),
        }
    }

    fn insert(
        &self,
        root_actor_id: ActorId,
        root_actor_cell: WeakInstanceCell,
    ) -> Result<(), anyhow::Error> {
        match self.roots.insert(root_actor_id.clone(), root_actor_cell) {
            None => Ok(()),
            // This should never happen because we do not recycle root actor's
            // IDs.
            Some(current_cell) => {
                let debugging_msg = match current_cell.upgrade() {
                    Some(cell) => format!("the stored cell's actor ID is {}", cell.actor_id()),
                    None => "the stored cell has been dropped".to_string(),
                };

                Err(anyhow::anyhow!(
                    "actor '{root_actor_id}' has already been added to ledger: {debugging_msg}"
                ))
            }
        }
    }

    /// Get a snapshot view of this ledger.
    fn snapshot(&self) -> ActorLedgerSnapshot {
        let roots = self
            .roots
            .iter()
            .flat_map(|r| {
                let (actor_id, weak_cell) = r.pair();
                // The actor might have been stopped or errored out. Since we do
                // not remove inactive actors from ledger, the upgrade() call
                // will return None in that scenario.
                weak_cell
                    .upgrade()
                    .map(|cell| (actor_id.clone(), Self::get_actor_tree_snapshot(&cell)))
            })
            .collect();

        ActorLedgerSnapshot { roots }
    }

    fn get_actor_tree_snapshot(cell: &InstanceCell) -> ActorTreeSnapshot {
        // Get the edges between this actor and its children.
        let children = cell
            .child_iter()
            .map(|child| (child.pid(), Self::get_actor_tree_snapshot(child.value())))
            .collect();

        ActorTreeSnapshot {
            pid: cell.actor_id().pid(),
            type_name: cell.state.actor_type.type_name().to_string(),
            status: cell.status().borrow().clone(),
            stats: ActorStats {
                num_processed_messages: cell.state.num_processed_messages.load(Ordering::SeqCst),
            },
            handlers: cell
                .state
                .exported_named_ports
                .iter()
                .map(|entry| (*entry.key(), entry.value().to_string()))
                .collect(),
            children,
            events: cell
                .state
                .recording
                .tail()
                .into_iter()
                .map(Event::from)
                .collect(),
            spans: cell
                .state
                .recording
                .stacks()
                .into_iter()
                .map(|stack| {
                    stack
                        .into_iter()
                        .map(|meta| meta.name().to_string())
                        .collect()
                })
                .collect(),
        }
    }
}

impl Proc {
    /// Create a new proc with the given proc id and forwarder.
    pub fn new(proc_id: ProcId, forwarder: BoxedMailboxSender) -> Self {
        Self::new_with_clock(proc_id, forwarder, ClockKind::default())
    }

    /// Create a new proc with the given proc id, forwarder and clock kind.
    pub fn new_with_clock(
        proc_id: ProcId,
        forwarder: BoxedMailboxSender,
        clock: ClockKind,
    ) -> Self {
        Self(Arc::new(ProcState {
            proc_id,
            proc_muxer: MailboxMuxer::new(),
            forwarder,
            roots: DashMap::new(),
            ledger: ActorLedger::new(),
            supervision_coordinator_port: OnceLock::new(),
            clock,
        }))
    }

    /// Set the supervision coordinator's port for this proc. Return Err if it is
    /// already set.
    pub fn set_supervision_coordinator(
        &self,
        port: PortHandle<ActorSupervisionEvent>,
    ) -> Result<(), anyhow::Error> {
        self.state()
            .supervision_coordinator_port
            .set(port)
            .map_err(|existing| anyhow::anyhow!("coordinator port is already set to {existing}"))
    }

    fn handle_supervision_event(&self, event: ActorSupervisionEvent) {
        let result = match self.state().supervision_coordinator_port.get() {
            Some(port) => port.send(event).map_err(anyhow::Error::from),
            None => Err(anyhow::anyhow!(
                "coordinator port is not set for proc {}",
                self.proc_id()
            )),
        };
        if let Err(err) = result {
            tracing::error!(
                "proc {}: could not propagate supervision event: {:?}: crashing",
                self.proc_id(),
                err
            );

            std::process::exit(1);
        }
    }

    /// Create a new local-only proc. This proc is not allowed to forward messages
    /// outside of the proc itself.
    pub fn local() -> Self {
        // TODO: name these something that is ~ globally unique, e.g., incorporate
        // the hostname, some GUID, etc.
        let proc_id = ProcId(id!(local), NEXT_LOCAL_RANK.fetch_add(1, Ordering::Relaxed));
        // TODO: make it so that local procs can talk to each other.
        Proc::new(proc_id, BoxedMailboxSender::new(PanickingMailboxSender))
    }

    /// The proc's ID.
    pub fn proc_id(&self) -> &ProcId {
        &self.state().proc_id
    }

    /// Shared sender used by the proc to forward messages to remote
    /// destinations.
    pub fn forwarder(&self) -> &BoxedMailboxSender {
        &self.0.forwarder
    }

    /// Convenience accessor for state.
    fn state(&self) -> &ProcState {
        self.0.as_ref()
    }

    /// The proc's clock.
    pub fn clock(&self) -> &ClockKind {
        &self.state().clock
    }

    /// Get the snapshot of the ledger.
    pub fn ledger_snapshot(&self) -> ActorLedgerSnapshot {
        self.state().ledger.snapshot()
    }

    /// Attach a mailbox to the proc with the provided root name.
    pub fn attach(&self, name: &str) -> Result<Mailbox, anyhow::Error> {
        let actor_id: ActorId = self.allocate_root_id(name)?;
        Ok(self.bind_mailbox(actor_id))
    }

    /// Attach a mailbox to the proc as a child actor.
    pub fn attach_child(&self, parent_id: &ActorId) -> Result<Mailbox, anyhow::Error> {
        let actor_id: ActorId = self.allocate_child_id(parent_id)?;
        Ok(self.bind_mailbox(actor_id))
    }

    /// Bind a mailbox to the proc.
    fn bind_mailbox(&self, actor_id: ActorId) -> Mailbox {
        let mbox = Mailbox::new(actor_id, BoxedMailboxSender::new(self.downgrade()));

        // TODO: T210748165 tie the muxer entry to the lifecycle of the mailbox held
        // by the caller. This will likely require a weak reference.
        self.state().proc_muxer.bind_mailbox(mbox.clone());
        mbox
    }

    /// Attach a mailbox to the proc with the provided root name, and bind an [`ActorRef`].
    /// This is intended only for testing, and will be replaced by simpled utilities.
    pub fn attach_actor<R, M>(
        &self,
        name: &str,
    ) -> Result<(Mailbox, ActorRef<R>, PortReceiver<M>), anyhow::Error>
    where
        M: RemoteMessage,
        R: RemoteActor + RemoteHandles<M>,
    {
        let mbox = self.attach(name)?;
        let (handle, rx) = mbox.open_port::<M>();
        handle.bind_to(M::port());
        let actor_ref = ActorRef::attest(mbox.actor_id().clone());
        Ok((mbox, actor_ref, rx))
    }

    /// Spawn a named (root) actor on this proc. The name of the actor must be
    /// unique.
    pub async fn spawn<A: Actor>(
        &self,
        name: &str,
        params: A::Params,
    ) -> Result<ActorHandle<A>, anyhow::Error> {
        let actor_id = self.allocate_root_id(name)?;
        let _ = tracing::debug_span!(
            "spawn_actor",
            actor_name = name,
            actor_type = std::any::type_name::<A>(),
            actor_id = actor_id.to_string(),
        );
        let instance = Instance::new(self.clone(), actor_id.clone(), None);
        let actor = A::new(params).await?;
        // Add this actor to the proc's actor ledger. We do not actively remove
        // inactive actors from ledger, because the actor's state can be inferred
        // from its weak cell.
        self.state()
            .ledger
            .insert(actor_id.clone(), instance.cell.downgrade())?;

        instance.start(actor).await
    }

    /// Spawn a child actor from the provided parent on this proc. The parent actor
    /// must already belong to this proc, a fact which is asserted in code.
    ///
    /// When spawn_child returns, the child has an associated cell and is linked
    /// with its parent.
    async fn spawn_child<A: Actor>(
        &self,
        parent: InstanceCell,
        params: A::Params,
    ) -> Result<ActorHandle<A>, anyhow::Error> {
        let actor_id = self.allocate_child_id(parent.actor_id())?;
        let instance = Instance::new(self.clone(), actor_id, Some(parent.clone()));
        let actor = A::new(params).await?;
        instance.start(actor).await
    }

    /// Call `abort` on the `JoinHandle` associated with the given
    /// root actor. If successful return `Some(root.clone())` else
    /// `None`.
    pub fn abort_root_actor(&mut self, root: &ActorId) -> Option<ActorId> {
        self.state()
            .ledger
            .roots
            .get(root)
            .into_iter()
            .flat_map(|e| e.upgrade())
            .map(|cell| {
                // `start` was called on the actor's instance
                // immediately following `root`'s insertion into the
                // ledger. Since `Instance::start()` is infallible we
                // know the cell's task handle has been set so it's
                // safe to `unwrap`.
                let h = cell.actor_task_handle().unwrap();
                tracing::debug!("{}: aborting {:?}", root, h);
                h.abort();
                root.clone()
            })
            .next()
    }

    // Iterating over a proc's root actors signaling each to stop.
    // Return the root actor IDs and status observers.
    async fn destroy(
        &mut self,
    ) -> Result<HashMap<ActorId, watch::Receiver<ActorStatus>>, anyhow::Error> {
        tracing::debug!("{}: proc stopping", self.proc_id());

        let mut statuses = HashMap::new();
        for entry in self.state().ledger.roots.iter() {
            match entry.value().upgrade() {
                None => (), // the root's cell has been dropped
                Some(cell) => {
                    if let Err(err) = cell.signal(Signal::DrainAndStop) {
                        tracing::error!(
                            "{}: failed to send stop signal to pid {}: {:?}",
                            self.proc_id(),
                            cell.pid(),
                            err
                        );
                        continue;
                    }
                    statuses.insert(cell.actor_id().clone(), cell.status().clone());
                }
            }
        }

        tracing::debug!("{}: proc stopped", self.proc_id());
        Ok(statuses)
    }

    /// Stop the proc. Returns a pair of:
    /// - the actors observed to stop;
    /// - the actors not observed to stop when timeout.
    ///
    /// The "skip_waiting" actor, if it is Some, is always not observed to stop.
    #[hyperactor::instrument]
    pub async fn destroy_and_wait(
        &mut self,
        timeout: Duration,
        skip_waiting: Option<&ActorId>,
    ) -> Result<(Vec<ActorId>, Vec<ActorId>), anyhow::Error> {
        let mut statuses = self.destroy().await?;
        let waits: Vec<_> = statuses
            .iter_mut()
            .filter(|(actor_id, _)| Some(*actor_id) != skip_waiting)
            .map(|(actor_id, root)| {
                let actor_id = actor_id.clone();
                async move {
                    tokio::time::timeout(
                        timeout,
                        root.wait_for(|state: &ActorStatus| matches!(*state, ActorStatus::Stopped)),
                    )
                    .await
                    .ok()
                    .map(|_| actor_id)
                }
            })
            .collect();

        let results = futures::future::join_all(waits).await;
        let stopped_actors: Vec<_> = results
            .iter()
            .filter_map(|actor_id| actor_id.as_ref())
            .cloned()
            .collect();
        let aborted_actors: Vec<_> = statuses
            .iter()
            .filter(|(actor_id, _)| !stopped_actors.contains(actor_id))
            .map(|(actor_id, _)| {
                let _: Option<ActorId> = self.abort_root_actor(actor_id);
                // If `is_none(&_)` then the proc's `ledger.roots`
                // contains an entry that wasn't a root or, the
                // associated actor's instance cell was already
                // dropped when we went to call `abort()` on the
                // cell's task handle.

                actor_id.clone()
            })
            .collect();

        tracing::info!(
            "destroy_and_wait: {} actors stopped, {} actors aborted",
            stopped_actors.len(),
            aborted_actors.len()
        );
        Ok((stopped_actors, aborted_actors))
    }

    /// Create a root allocation in the proc.
    #[hyperactor::instrument]
    fn allocate_root_id(&self, name: &str) -> Result<ActorId, anyhow::Error> {
        let name = name.to_string();
        match self.state().roots.entry(name.to_string()) {
            Entry::Vacant(entry) => {
                entry.insert(AtomicUsize::new(1));
            }
            Entry::Occupied(_) => {
                anyhow::bail!("an actor with name '{}' has already been spawned", name)
            }
        }
        Ok(ActorId(self.state().proc_id.clone(), name.to_string(), 0))
    }

    /// Create a child allocation in the proc.
    pub(crate) fn allocate_child_id(&self, parent_id: &ActorId) -> Result<ActorId, anyhow::Error> {
        assert_eq!(*parent_id.proc_id(), self.state().proc_id);
        let pid = match self.state().roots.get(parent_id.name()) {
            None => anyhow::bail!(
                "no actor named {} in proc {}",
                parent_id.name(),
                self.state().proc_id
            ),
            Some(next_pid) => next_pid.fetch_add(1, Ordering::Relaxed),
        };
        Ok(parent_id.child_id(pid))
    }

    fn downgrade(&self) -> WeakProc {
        WeakProc::new(self)
    }
}

#[async_trait]
impl MailboxSender for Proc {
    fn post(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        if envelope.dest().actor_id().proc_id() == &self.state().proc_id {
            self.state().proc_muxer.post(envelope, return_handle)
        } else {
            self.state().forwarder.post(envelope, return_handle)
        }
    }
}

#[derive(Debug)]
struct WeakProc(Weak<ProcState>);

impl WeakProc {
    fn new(proc: &Proc) -> Self {
        Self(Arc::downgrade(&proc.0))
    }

    fn upgrade(&self) -> Option<Proc> {
        self.0.upgrade().map(Proc)
    }
}

#[async_trait]
impl MailboxSender for WeakProc {
    fn post(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        match self.upgrade() {
            Some(proc) => proc.post(envelope, return_handle),
            None => envelope.undeliverable(
                DeliveryError::BrokenLink("fail to upgrade WeakProc".to_string()),
                return_handle,
            ),
        }
    }
}

/// Represents a single work item used by the instance to dispatch to
/// actor handles. Specifically, this enables handler polymorphism.
struct WorkCell<A: Actor + Send>(
    Box<
        dyn for<'a> FnOnce(
                &'a mut A,
                &'a Instance<A>,
            )
                -> Pin<Box<dyn Future<Output = Result<(), anyhow::Error>> + 'a + Send>>
            + Send
            + Sync,
    >,
);

impl<A: Actor + Send> WorkCell<A> {
    /// Create a new WorkCell from a concrete function (closure).
    fn new(
        f: impl for<'a> FnOnce(
            &'a mut A,
            &'a Instance<A>,
        )
            -> Pin<Box<dyn Future<Output = Result<(), anyhow::Error>> + 'a + Send>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self(Box::new(f))
    }

    /// Handle the message represented by this work cell.
    fn handle<'a>(
        self,
        actor: &'a mut A,
        instance: &'a Instance<A>,
    ) -> Pin<Box<dyn Future<Output = Result<(), anyhow::Error>> + Send + 'a>> {
        (self.0)(actor, instance)
    }
}

/// An actor instance. This is responsible for managing a running actor, including
/// its full lifecycle, supervision, signal management, etc. Instances can represent
/// a managed actor or a "client" actor that has joined the proc.
pub struct Instance<A: Actor> {
    /// The proc that owns this instance.
    proc: Proc,

    /// The instance cell that manages instance hierarchy.
    cell: InstanceCell,

    /// The mailbox associated with the actor.
    mailbox: Mailbox,

    /// The actor's signal receiver. This is used to
    /// receive signals sent to the actor.
    signal_receiver: PortReceiver<Signal>,

    /// The actor's supervision event receiver. This is used to receive supervision
    /// event from the parent.
    supervision_event_receiver: PortReceiver<ActorSupervisionEvent>,

    ports: Arc<Ports<A>>,

    /// Handler work queue.
    work_rx: mpsc::UnboundedReceiver<WorkCell<A>>,

    /// A watch for communicating the actor's state.
    status_tx: watch::Sender<ActorStatus>,

    status_span: Mutex<tracing::Span>,

    /// The timestamp of when the currently active status was set.
    _last_status_change: Arc<tokio::time::Instant>,
}

impl<A: Actor> Instance<A> {
    /// Create a new actor instance in Created state.
    pub(crate) fn new(proc: Proc, actor_id: ActorId, parent: Option<InstanceCell>) -> Self {
        // Set up messaging
        let mailbox = Mailbox::new(actor_id.clone(), BoxedMailboxSender::new(proc.downgrade()));
        let (work_tx, work_rx) = mpsc::unbounded_channel();

        let ports: Arc<Ports<A>> = Arc::new(Ports::new(mailbox.clone(), work_tx));
        let (signal_port, signal_receiver) = ports.open_message_port().unwrap();

        proc.state().proc_muxer.bind_mailbox(mailbox.clone());

        // Supervision event port is used locally in the proc
        let (supervision_port, supervision_event_receiver) = mailbox.open_port();

        let (status_tx, status_rx) = watch::channel(ActorStatus::Created);

        let actor_type = match TypeInfo::of::<A>() {
            Some(info) => ActorType::Named(info),
            None => ActorType::Anonymous(std::any::type_name::<A>()),
        };
        let ais = actor_id.to_string();

        let cell = InstanceCell::new(
            actor_id,
            actor_type,
            signal_port,
            supervision_port,
            status_rx,
            parent,
        );
        let start = proc.clock().now();

        Self {
            proc,
            cell,
            mailbox,
            signal_receiver,
            supervision_event_receiver,
            ports,
            work_rx,
            status_tx,
            status_span: Mutex::new(tracing::debug_span!(
                "actor_status",
                actor_id = ais,
                name = "created"
            )),
            _last_status_change: Arc::new(start),
        }
    }

    /// Notify subscribers of a change in the actors status and bump counters with the duration which
    /// the last status was active for.
    fn change_status(&self, new: ActorStatus) {
        // let old = self.status_tx.send_replace(new.clone());
        self.status_tx.send_replace(new.clone());
        let actor_id_str = self.self_id().to_string();
        *self.status_span.lock().expect("can't change") = tracing::debug_span!(
            "actor_status",
            actor_id = actor_id_str,
            name = new.arm().unwrap_or_default()
        );
    }

    /// This instance's actor ID.
    pub fn self_id(&self) -> &ActorId {
        self.mailbox.actor_id()
    }

    /// Signal the actor to stop.
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `ActorError`.
    pub fn stop(&self) -> Result<(), ActorError> {
        self.cell.signal(Signal::DrainAndStop)
    }

    /// Open a new port that accepts M-typed messages. The returned
    /// port may be freely cloned, serialized, and passed around. The
    /// returned receiver should only be retained by the actor responsible
    /// for processing the delivered messages.
    pub fn open_port<M: Message>(&self) -> (PortHandle<M>, PortReceiver<M>) {
        self.mailbox.open_port()
    }

    /// Open a new one-shot port that accepts M-typed messages. The
    /// returned port may be used to send a single message; ditto the
    /// receiver may receive a single message.
    pub fn open_once_port<M: Message>(&self) -> (OncePortHandle<M>, OncePortReceiver<M>) {
        self.mailbox.open_once_port()
    }

    /// Send a message to the actor running on the proc.
    pub fn post(&self, port_id: PortId, message: Serialized) {
        <Self as cap::sealed::CanSend>::post(self, port_id, message)
    }

    /// Send a message to the actor itself with a delay usually to trigger some event.
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `ActorError`.
    pub fn self_message_with_delay<M>(&self, message: M, delay: Duration) -> Result<(), ActorError>
    where
        M: Message,
        A: Handler<M>,
    {
        let port = self.port();
        let self_id = self.self_id().clone();
        let clock = self.proc.state().clock.clone();
        tokio::spawn(async move {
            clock.non_advancing_sleep(delay).await;
            if let Err(e) = port.send(message) {
                // TODO: this is a fire-n-forget thread. We need to
                // handle errors in a better way.
                tracing::info!("{}: error sending delayed message: {}", self_id, e);
            }
        });
        Ok(())
    }

    /// Start an A-typed actor onto this instance with the provided params. When spawn returns,
    /// the actor has been linked with its parent, if it has one.
    #[hyperactor::instrument]
    async fn start(self, actor: A) -> Result<ActorHandle<A>, anyhow::Error> {
        let instance_cell = self.cell.clone();
        let actor_id = self.cell.actor_id().clone();
        let actor_handle = ActorHandle::new(self.cell.clone(), self.ports.clone());
        let actor_task_handle =
            A::spawn_server_task(panic_handler::with_backtrace_tracking(self.serve(actor)));
        tracing::debug!("{}: spawned with {:?}", actor_id, actor_task_handle);
        instance_cell
            .state
            .actor_task_handle
            .set(actor_task_handle)
            .unwrap_or_else(|_| panic!("{}: task handle store failed", actor_id));

        Ok(actor_handle)
    }

    async fn serve(mut self, mut actor: A) {
        let result = self.run_actor_tree(&mut actor).await;

        let actor_status = match result {
            Ok(_) => ActorStatus::Stopped,
            Err(err) => ActorStatus::Failed(err.to_string()),
        };

        let result = self.cell.maybe_unlink_parent();
        if let Some(parent) = result {
            if let Err(err) = parent.signal(Signal::ChildStopped(self.cell.pid())) {
                tracing::error!(
                    "{}: failed to send stop message to parent pid {}: {:?}",
                    self.self_id(),
                    parent.pid(),
                    err
                );
            }
            if actor_status.is_failed() {
                // Parent exists, failure should be propagated to the parent.
                parent.send_supervision_event_or_crash(ActorSupervisionEvent::new(
                    self.cell.actor_id().clone(),
                    actor_status.clone(),
                ));
            }
        } else {
            // Failure happened to the root actor or orphaned child actors.
            // In either case, the failure should be propagated to proc.
            //
            // Note that orphaned actor is unexpected and would only happen if
            // there is a bug.
            if actor_status.is_failed() {
                self.proc
                    .handle_supervision_event(ActorSupervisionEvent::new(
                        self.cell.actor_id().clone(),
                        actor_status.clone(),
                    ))
            }
        }
        self.change_status(actor_status);
    }

    /// Runs the actor, and manages its supervision tree. When the function returns,
    /// the whole tree rooted at this actor has stopped.
    async fn run_actor_tree(&mut self, actor: &mut A) -> Result<(), ActorError> {
        // It is okay to catch all panics here, because we are in a tokio task,
        // and tokio will catch the panic anyway:
        // https://docs.rs/tokio/latest/tokio/task/struct.JoinError.html#method.is_panic
        // What we do here is just to catch it early so we can handle it.

        let result = match AssertUnwindSafe(self.run(actor)).catch_unwind().await {
            Ok(result) => result,
            Err(err) => {
                // This is only the error message. Backtrace is not included.
                let err_msg = err
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| err.downcast_ref::<String>().map(|s| s.as_str()))
                    .unwrap_or("panic cannot be downcasted");

                let backtrace = panic_handler::take_panic_backtrace()
                    .unwrap_or_else(|e| format!("Cannot take backtrace due to: {:?}", e));
                Err(ActorError::new(
                    self.self_id().clone(),
                    ActorErrorKind::Panic(anyhow::anyhow!("{}\n{}", err_msg, backtrace)),
                ))
            }
        };

        if let Err(ref err) = result {
            tracing::error!("{}: actor failure: {}", self.self_id(), err);
        }
        self.change_status(ActorStatus::Stopping);

        // After this point, we know we won't spawn any more children,
        // so we can safely read the current child keys.
        let mut to_unlink = Vec::new();
        for child in self.cell.child_iter() {
            if let Err(err) = child.value().signal(Signal::Stop) {
                tracing::error!(
                    "{}: failed to send stop signal to child pid {}: {:?}",
                    self.self_id(),
                    child.key(),
                    err
                );
                to_unlink.push(child.value().clone());
            }
        }
        // Manually unlink children that have already been stopped.
        for child in to_unlink {
            self.cell.unlink(&child);
        }

        while self.cell.child_count() > 0 {
            match self.signal_receiver.recv().await? {
                Signal::ChildStopped(pid) => {
                    if let Some(child) = self.cell.get_child(pid) {
                        self.cell.unlink(&child);
                    }
                }
                _ => (),
            }
        }

        result
    }

    /// Initialize and run the actor until it fails or is stopped.
    async fn run(&mut self, actor: &mut A) -> Result<(), ActorError> {
        hyperactor_telemetry::declare_static_counter!(MESSAGES_RECEIVED, "actor.messages_received");
        tracing::debug!("entering actor loop");

        self.change_status(ActorStatus::Initializing);
        actor
            .init(self)
            .await
            .map_err(|err| ActorError::new(self.self_id().clone(), ActorErrorKind::Init(err)))?;
        let need_drain;
        'messages: loop {
            self.change_status(ActorStatus::Idle);
            let metric_pairs =
                hyperactor_telemetry::kv_pairs!("actor_id" => self.self_id().to_string());
            tokio::select! {
                work = self.work_rx.recv() => {
                    MESSAGES_RECEIVED.add(1, metric_pairs);
                    MESSAGE_QUEUE_SIZE.add(-1, metric_pairs);
                    let _ = MESSAGE_HANDLER_DURATION.start(metric_pairs);
                    let work = work.expect("inconsistent work queue state");
                    if let Err(err) = work.handle(actor, self).await {
                        for supervision_event in self.supervision_event_receiver.drain() {
                            self.handle_supervision_event(actor, supervision_event).await;
                        }
                        return Err(ActorError::new(self.self_id().clone(), ActorErrorKind::Processing(err)));
                    }
                }
                signal = self.signal_receiver.recv() => {
                    let signal = signal.map_err(ActorError::from);
                    match signal? {
                        signal@(Signal::Stop | Signal::DrainAndStop) => {
                            need_drain = matches!(signal, Signal::DrainAndStop);
                            break 'messages;
                        },
                        Signal::ChildStopped(pid) => {
                            let result = self.cell.get_child(pid);
                            if let Some(child) = result {
                                self.cell.unlink(&child);
                            } else {
                                tracing::warn!("received signal for unknown child pid {}", pid);
                            }
                        },
                    }
                }
                Ok(supervision_event) = self.supervision_event_receiver.recv() => {
                    self.handle_supervision_event(actor, supervision_event).await;
                }
            }
            self.cell
                .state
                .num_processed_messages
                .fetch_add(1, Ordering::SeqCst);
        }

        if need_drain {
            self.change_status(ActorStatus::Stopping);
            let mut n = 0;
            while let Ok(work) = self.work_rx.try_recv() {
                if let Err(err) = work.handle(actor, self).await {
                    return Err(ActorError::new(
                        self.self_id().clone(),
                        ActorErrorKind::Processing(err),
                    ));
                }
                n += 1;
            }
            tracing::debug!("drained {} messages", n);
        }
        tracing::debug!("exited actor loop");
        self.change_status(ActorStatus::Stopped);
        Ok(())
    }

    async fn handle_supervision_event(
        &self,
        actor: &mut A,
        supervision_event: ActorSupervisionEvent,
    ) {
        // Handle the supervision event with the current actor.
        if let Ok(false) = actor
            .handle_supervision_event(self, &supervision_event)
            .await
        {
            // The supervision event wasn't handled by this actor, try to bubble it up.
            let result = self.cell.get_parent_cell();
            if let Some(parent) = result {
                parent.send_supervision_event_or_crash(supervision_event);
            } else {
                // Reaching here means the actor is either a root actor, or an orphaned
                // child actor (i.e. the parent actor was dropped unexpectedly). In either
                // case, the supervision event should be sent to proc.
                //
                // Note that orphaned actor is unexpected and would only happen if there
                // is a bug.
                self.proc.handle_supervision_event(supervision_event);
            }
        }
    }

    async unsafe fn handle_message<M: Message>(
        &self,
        actor: &mut A,
        type_info: Option<&'static TypeInfo>,
        message: M,
    ) -> Result<(), anyhow::Error>
    where
        A: Handler<M>,
    {
        let handler = type_info.map(|info| {
            (
                info.typename().to_string(),
                // SAFETY: The caller promises to pass the correct type info.
                unsafe {
                    info.arm_unchecked(&message as *const M as *const ())
                        .map(str::to_string)
                },
            )
        });

        let _ = self.change_status(ActorStatus::Processing(
            self.clock().system_time_now(),
            handler,
        ));
        let span = self.status_span.lock().unwrap().clone();
        actor.handle(self, message).instrument(span).await
    }

    /// Return a handle port handle representing the actor's message
    /// handler for M-typed messages.
    pub fn port<M: Message>(&self) -> PortHandle<M>
    where
        A: Handler<M>,
    {
        self.ports.get()
    }

    /// The [`ActorHandle`] corresponding to this instance.
    pub fn handle(&self) -> ActorHandle<A> {
        ActorHandle::new(self.cell.clone(), Arc::clone(&self.ports))
    }

    /// The owning actor ref.
    pub fn bind<R: Binds<A>>(&self) -> ActorRef<R> {
        self.cell.bind(self.ports.as_ref())
    }

    // Temporary in order to support python bindings.
    #[doc(hidden)]
    pub fn mailbox_for_py(&self) -> &Mailbox {
        &self.mailbox
    }

    /// A reference to the proc's clock
    pub fn clock(&self) -> &(impl Clock + use<A>) {
        &self.proc.state().clock
    }

    /// The owning proc.
    pub fn proc(&self) -> &Proc {
        &self.proc
    }
}

impl<A: Actor> cap::sealed::CanSend for Instance<A> {
    fn post(&self, dest: PortId, data: Serialized) {
        let envelope = MessageEnvelope::new(self.self_id().clone(), dest, data);
        self.proc.post(envelope, self.ports.get());
    }
}

impl<A: Actor> cap::sealed::CanOpenPort for Instance<A> {
    fn mailbox(&self) -> &Mailbox {
        &self.mailbox
    }
}

impl<A: Actor> cap::sealed::CanSplitPort for Instance<A> {
    fn split(&self, port_id: PortId, reducer_typehash: Option<u64>) -> PortId {
        self.mailbox.split(port_id, reducer_typehash)
    }
}

#[async_trait]
impl<A: Actor> cap::sealed::CanSpawn for Instance<A> {
    async fn spawn<C: Actor>(&self, params: C::Params) -> anyhow::Result<ActorHandle<C>> {
        self.proc.spawn_child(self.cell.clone(), params).await
    }
}

#[derive(Debug)]
enum ActorType {
    Named(&'static TypeInfo),
    Anonymous(&'static str),
}

impl ActorType {
    /// The actor's type name.
    fn type_name(&self) -> &'static str {
        match self {
            Self::Named(info) => info.typename(),
            Self::Anonymous(name) => name,
        }
    }
}

/// InstanceCell contains all of the type-erased, shareable state of an instance.
/// Specifically, InstanceCells form a supervision tree, and is used by ActorHandle
/// to access the underlying instance.
///
/// InstanceCell is reference counted and cloneable.
#[derive(Clone, Debug)]
pub struct InstanceCell {
    state: Arc<InstanceState>,
}

#[derive(Debug)]
struct InstanceState {
    /// The actor's id.
    actor_id: ActorId,

    /// Actor info contains the actor's type information.
    actor_type: ActorType,

    /// The actor's signal port. This is used to send
    /// signals to the actor.
    signal: PortHandle<Signal>,

    /// The actor's supervision port. This is used to send
    /// supervision event to the actor.
    supervision_port: PortHandle<ActorSupervisionEvent>,

    /// An observer that stores the current status of the actor.
    status: watch::Receiver<ActorStatus>,

    /// A weak reference to this instance's parent.
    parent: WeakInstanceCell,

    /// This instance's children by their PIDs.
    children: DashMap<Index, InstanceCell>,

    /// Access to the spawned actor's join handle.
    actor_task_handle: OnceLock<JoinHandle<()>>,

    /// The set of named ports that are exported by this actor.
    exported_named_ports: DashMap<u64, &'static str>,

    /// The number of messages processed by this actor.
    num_processed_messages: AtomicU64,

    /// The log recording associated with this actor. It is used to
    /// store a 'flight record' of events while the actor is running.
    recording: Recording,
}

impl InstanceCell {
    /// Creates a new instance cell with the provided internal state. If a parent
    /// is provided, it is linked to this cell.
    fn new(
        actor_id: ActorId,
        actor_type: ActorType,
        signal: PortHandle<Signal>,
        supervision_port: PortHandle<ActorSupervisionEvent>,
        status: watch::Receiver<ActorStatus>,
        parent: Option<InstanceCell>,
    ) -> Self {
        let _ais = actor_id.to_string();
        let cell = Self {
            state: Arc::new(InstanceState {
                actor_id,
                actor_type,
                signal,
                supervision_port,
                status,
                parent: parent.map_or_else(WeakInstanceCell::new, |cell| cell.downgrade()),
                children: DashMap::new(),
                actor_task_handle: OnceLock::new(),
                exported_named_ports: DashMap::new(),
                num_processed_messages: AtomicU64::new(0),
                recording: hyperactor_telemetry::recorder().record(64),
            }),
        };
        cell.maybe_link_parent();
        cell
    }

    fn wrap(state: Arc<InstanceState>) -> Self {
        Self { state }
    }

    /// The actor's ID.
    pub(crate) fn actor_id(&self) -> &ActorId {
        &self.state.actor_id
    }

    /// The actor's PID.
    pub(crate) fn pid(&self) -> Index {
        self.state.actor_id.pid()
    }

    /// The actor's join handle.
    pub(crate) fn actor_task_handle(&self) -> Option<&JoinHandle<()>> {
        self.state.actor_task_handle.get()
    }

    /// The instance's status observer.
    pub(crate) fn status(&self) -> &watch::Receiver<ActorStatus> {
        &self.state.status
    }

    /// Send a signal to the actor.
    #[allow(clippy::result_large_err)] // TODO: Consider reducing the size of `ActorError`.
    pub fn signal(&self, signal: Signal) -> Result<(), ActorError> {
        self.state.signal.send(signal).map_err(ActorError::from)
    }

    /// Used by this actor's children to send a supervision event to this actor.
    /// When it fails to send, we will crash the process. As part of the crash,
    /// all the procs and actors running on this process will be terminated
    /// forcefully.
    ///
    /// Note that "let it crash" is the default behavior when a supervision event
    /// cannot be delivered upstream. It is the upstream's responsibility to
    /// detect and handle crashes.
    pub fn send_supervision_event_or_crash(&self, event: ActorSupervisionEvent) {
        if let Err(err) = self.state.supervision_port.send(event) {
            tracing::error!(
                "{}: failed to send supervision event to actor: {:?}. Crash the process.",
                self.actor_id(),
                err
            );

            std::process::exit(1);
        }
    }

    /// Downgrade this InstanceCell to a weak reference.
    fn downgrade(&self) -> WeakInstanceCell {
        WeakInstanceCell {
            state: Arc::downgrade(&self.state),
        }
    }

    /// Link this instance to a new child.
    fn link(&self, child: InstanceCell) {
        assert_eq!(self.actor_id().proc_id(), child.actor_id().proc_id());
        self.state.children.insert(child.pid(), child);
    }

    /// Unlink this instance from a child.
    fn unlink(&self, child: &InstanceCell) {
        assert_eq!(self.actor_id().proc_id(), child.actor_id().proc_id());
        self.state.children.remove(&child.pid());
    }

    /// Link this instance to its parent, if it has one.
    fn maybe_link_parent(&self) {
        if let Some(parent) = self.state.parent.upgrade() {
            parent.link(self.clone());
        }
    }

    /// Unlink this instance from its parent, if it has one. If it was unlinked,
    /// the parent is returned.
    fn maybe_unlink_parent(&self) -> Option<InstanceCell> {
        let result = self.state.parent.upgrade();
        if let Some(parent) = &result {
            parent.unlink(self);
        }
        result
    }

    /// Get parent instance cell, if it exists.
    fn get_parent_cell(&self) -> Option<InstanceCell> {
        self.state.parent.upgrade()
    }

    /// Return an iterator over this instance's children. This may deadlock if the
    /// caller already holds a reference to any item in map.
    fn child_iter(&self) -> impl Iterator<Item = RefMulti<'_, Index, InstanceCell>> {
        self.state.children.iter()
    }

    /// The number of children this instance has.
    fn child_count(&self) -> usize {
        self.state.children.len()
    }

    /// Get a child by its PID.
    fn get_child(&self, pid: Index) -> Option<InstanceCell> {
        self.state.children.get(&pid).map(|child| child.clone())
    }

    /// This is temporary so that we can share binding code between handle and instance.
    /// We should find some (better) way to consolidate the two.
    pub(crate) fn bind<A: Actor, R: Binds<A>>(&self, ports: &Ports<A>) -> ActorRef<R> {
        <R as Binds<A>>::bind(ports);
        // All actors handle signals:
        ports.bind::<Signal>();
        // TODO: consider sharing `ports.bound` directly.
        for entry in ports.bound.iter() {
            self.state
                .exported_named_ports
                .insert(*entry.key(), entry.value());
        }
        ActorRef::attest(self.actor_id().clone())
    }
}

/// A weak version of the InstanceCell. This is used to provide cyclical
/// linkage between actors without creating a strong reference cycle.
#[derive(Debug, Clone)]
struct WeakInstanceCell {
    state: Weak<InstanceState>,
}

impl WeakInstanceCell {
    /// Create a new weak instance cell that is never upgradeable.
    fn new() -> Self {
        Self { state: Weak::new() }
    }

    /// Upgrade this weak instance cell to a strong reference, if possible.
    fn upgrade(&self) -> Option<InstanceCell> {
        self.state.upgrade().map(InstanceCell::wrap)
    }
}

/// A polymorphic dictionary that stores ports for an actor's handlers.
/// The interface memoizes the ports so that they are reused. We do not
/// (yet) support stable identifiers across multiple instances of the same
/// actor.
pub struct Ports<A: Actor> {
    ports: DashMap<TypeId, Box<dyn Any + Send + Sync + 'static>>,
    bound: DashMap<u64, &'static str>,
    mailbox: Mailbox,
    workq: mpsc::UnboundedSender<WorkCell<A>>,
}

impl<A: Actor> Ports<A> {
    fn new(mailbox: Mailbox, workq: mpsc::UnboundedSender<WorkCell<A>>) -> Self {
        Self {
            ports: DashMap::new(),
            bound: DashMap::new(),
            mailbox,
            workq,
        }
    }

    /// Get a port for the Handler<M> of actor A.
    pub(crate) fn get<M: Message>(&self) -> PortHandle<M>
    where
        A: Handler<M>,
    {
        let key = TypeId::of::<M>();
        match self.ports.entry(key) {
            Entry::Vacant(entry) => {
                // Some special case hackery, but it keeps the rest of the code (relatively) simple.
                assert_ne!(
                    key,
                    TypeId::of::<Signal>(),
                    "cannot provision Signal port through `Ports::get`"
                );

                let type_info = TypeInfo::get_by_typeid(key);
                let workq = self.workq.clone();
                let actor_id = self.mailbox.actor_id().to_string();
                let port = self.mailbox.open_enqueue_port(move |msg: M| {
                    let work = WorkCell::new(move |actor: &mut A, instance: &Instance<A>| {
                        Box::pin(async move {
                            // SAFETY: we guarantee that the passed type_info is for type M.
                            unsafe { instance.handle_message(actor, type_info, msg).await }
                        })
                    });
                    MESSAGE_QUEUE_SIZE.add(
                        1,
                        hyperactor_telemetry::kv_pairs!("actor_id" => actor_id.clone()),
                    );
                    workq.send(work).map_err(anyhow::Error::from)
                });
                entry.insert(Box::new(port.clone()));
                port
            }
            Entry::Occupied(entry) => {
                let port = entry.get();
                port.downcast_ref::<PortHandle<M>>().unwrap().clone()
            }
        }
    }

    /// Open a (typed) message port as in [`get`], but return a port receiver instead of dispatching
    /// the underlying handler.
    pub(crate) fn open_message_port<M: Message>(&self) -> Option<(PortHandle<M>, PortReceiver<M>)> {
        match self.ports.entry(TypeId::of::<M>()) {
            Entry::Vacant(entry) => {
                let (port, receiver) = self.mailbox.open_port();
                entry.insert(Box::new(port.clone()));
                Some((port, receiver))
            }
            Entry::Occupied(_) => None,
        }
    }

    /// Bind the given message type to its default port.
    pub fn bind<M: RemoteMessage>(&self)
    where
        A: Handler<M>,
    {
        self.bind_to::<M>(M::port());
    }

    /// Bind the given message type to the provided port.
    /// Ports cannot be rebound to different message types;
    /// and attempting to do so will result in a panic.
    pub fn bind_to<M: RemoteMessage>(&self, port_index: u64)
    where
        A: Handler<M>,
    {
        match self.bound.entry(port_index) {
            Entry::Vacant(entry) => {
                self.get::<M>().bind_to(port_index);
                entry.insert(M::typename());
            }
            Entry::Occupied(entry) => {
                assert_eq!(
                    *entry.get(),
                    M::typename(),
                    "bind {}: port index {} already bound to type {}",
                    M::typename(),
                    port_index,
                    entry.get(),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;
    use std::sync::atomic::AtomicBool;

    use maplit::hashmap;
    use serde_json::json;
    use tokio::sync::Barrier;
    use tokio::sync::oneshot;
    use tracing::Level;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_test::internal::logs_with_scope_contain;

    use super::*;
    // needed for in-crate macro expansion
    use crate as hyperactor;
    use crate::HandleClient;
    use crate::Handler;
    use crate::clock::RealClock;
    use crate::test_utils::proc_supervison::ProcSupervisionCoordinator;
    use crate::test_utils::process_assertion::assert_termination;

    impl ActorTreeSnapshot {
        #[allow(dead_code)]
        fn empty(pid: Index) -> Self {
            Self {
                pid,
                type_name: String::new(),
                status: ActorStatus::Idle,
                stats: ActorStats::default(),
                handlers: HashMap::new(),
                children: HashMap::new(),
                events: Vec::new(),
                spans: Vec::new(),
            }
        }

        fn empty_typed(pid: Index, type_name: String) -> Self {
            Self {
                pid,
                type_name,
                status: ActorStatus::Idle,
                stats: ActorStats::default(),
                handlers: HashMap::new(),
                children: HashMap::new(),
                events: Vec::new(),
                spans: Vec::new(),
            }
        }
    }

    #[derive(Debug)]
    struct TestActor;

    #[derive(Handler, HandleClient, Debug)]
    enum TestActorMessage {
        Reply(oneshot::Sender<()>),
        Wait(oneshot::Sender<()>, oneshot::Receiver<()>),
        Forward(ActorHandle<TestActor>, Box<TestActorMessage>),
        Noop(),
        Fail(anyhow::Error),
        Panic(String),
        Spawn(oneshot::Sender<ActorHandle<TestActor>>),
    }

    impl TestActor {
        async fn spawn_child(parent: &ActorHandle<TestActor>) -> ActorHandle<TestActor> {
            let (tx, rx) = oneshot::channel();
            parent.send(TestActorMessage::Spawn(tx)).unwrap();
            rx.await.unwrap()
        }
    }

    #[async_trait]
    impl Actor for TestActor {
        type Params = ();

        async fn new(_params: ()) -> Result<Self, anyhow::Error> {
            Ok(Self)
        }
    }

    #[async_trait]
    #[crate::forward(TestActorMessage)]
    impl TestActorMessageHandler for TestActor {
        async fn reply(
            &mut self,
            _this: &Instance<Self>,
            sender: oneshot::Sender<()>,
        ) -> Result<(), anyhow::Error> {
            sender.send(()).unwrap();
            Ok(())
        }

        async fn wait(
            &mut self,
            _this: &Instance<Self>,
            sender: oneshot::Sender<()>,
            receiver: oneshot::Receiver<()>,
        ) -> Result<(), anyhow::Error> {
            sender.send(()).unwrap();
            receiver.await.unwrap();
            Ok(())
        }

        async fn forward(
            &mut self,
            _this: &Instance<Self>,
            destination: ActorHandle<TestActor>,
            message: Box<TestActorMessage>,
        ) -> Result<(), anyhow::Error> {
            // TODO: this needn't be async
            destination.send(*message)?;
            Ok(())
        }

        async fn noop(&mut self, _this: &Instance<Self>) -> Result<(), anyhow::Error> {
            Ok(())
        }

        async fn fail(
            &mut self,
            _this: &Instance<Self>,
            err: anyhow::Error,
        ) -> Result<(), anyhow::Error> {
            Err(err)
        }

        async fn panic(
            &mut self,
            _this: &Instance<Self>,
            err_msg: String,
        ) -> Result<(), anyhow::Error> {
            panic!("{}", err_msg);
        }

        async fn spawn(
            &mut self,
            this: &Instance<Self>,
            reply: oneshot::Sender<ActorHandle<TestActor>>,
        ) -> Result<(), anyhow::Error> {
            let handle = <Self as Actor>::spawn(this, ()).await?;
            reply.send(handle).unwrap();
            Ok(())
        }
    }

    #[tracing_test::traced_test]
    #[tokio::test]
    async fn test_spawn_actor() {
        let proc = Proc::local();
        let handle = proc.spawn::<TestActor>("test", ()).await.unwrap();

        // Check on the join handle.
        assert!(logs_contain(
            format!(
                "{}: spawned with {:?}",
                handle.actor_id(),
                handle.cell().actor_task_handle().unwrap(),
            )
            .as_str()
        ));

        let mut state = handle.status().clone();

        // Send a ping-pong to the actor. Wait for the actor to become idle.

        let (tx, rx) = oneshot::channel::<()>();
        handle.send(TestActorMessage::Reply(tx)).unwrap();
        rx.await.unwrap();

        state
            .wait_for(|state: &ActorStatus| matches!(*state, ActorStatus::Idle))
            .await
            .unwrap();

        // Make sure we enter processing state while the actor is handling a message.
        let (enter_tx, enter_rx) = oneshot::channel::<()>();
        let (exit_tx, exit_rx) = oneshot::channel::<()>();

        handle
            .send(TestActorMessage::Wait(enter_tx, exit_rx))
            .unwrap();
        enter_rx.await.unwrap();
        assert_matches!(*state.borrow(), ActorStatus::Processing(instant, _) if instant <= RealClock.system_time_now());
        exit_tx.send(()).unwrap();

        state
            .wait_for(|state| matches!(*state, ActorStatus::Idle))
            .await
            .unwrap();

        handle.drain_and_stop().unwrap();
        handle.await;
        assert_matches!(*state.borrow(), ActorStatus::Stopped);
    }

    #[tokio::test]
    async fn test_proc_actors_messaging() {
        let proc = Proc::local();
        let first = proc.spawn::<TestActor>("first", ()).await.unwrap();
        let second = proc.spawn::<TestActor>("second", ()).await.unwrap();
        let (tx, rx) = oneshot::channel::<()>();
        let reply_message = TestActorMessage::Reply(tx);
        first
            .send(TestActorMessage::Forward(second, Box::new(reply_message)))
            .unwrap();
        rx.await.unwrap();
    }

    fn validate_link(child: &InstanceCell, parent: &InstanceCell) {
        assert_eq!(child.actor_id().proc_id(), parent.actor_id().proc_id());
        assert_eq!(
            child.state.parent.upgrade().unwrap().actor_id(),
            parent.actor_id()
        );
        assert_matches!(
            parent.state.children.get(&child.pid()),
            Some(node) if node.actor_id() == child.actor_id()
        );
    }

    #[tracing_test::traced_test]
    #[tokio::test]
    async fn test_spawn_child() {
        let proc = Proc::local();

        let first = proc.spawn::<TestActor>("first", ()).await.unwrap();
        let second = TestActor::spawn_child(&first).await;
        let third = TestActor::spawn_child(&second).await;

        // Check we've got the join handles.
        assert!(logs_with_scope_contain(
            "hyperactor::proc",
            format!(
                "{}: spawned with {:?}",
                first.actor_id(),
                first.cell().actor_task_handle().unwrap()
            )
            .as_str()
        ));
        assert!(logs_with_scope_contain(
            "hyperactor::proc",
            format!(
                "{}: spawned with {:?}",
                second.actor_id(),
                second.cell().actor_task_handle().unwrap()
            )
            .as_str()
        ));
        assert!(logs_with_scope_contain(
            "hyperactor::proc",
            format!(
                "{}: spawned with {:?}",
                third.actor_id(),
                third.cell().actor_task_handle().unwrap()
            )
            .as_str()
        ));

        // These are allocated in sequence:
        assert_eq!(first.actor_id().proc_id(), proc.proc_id());
        assert_eq!(second.actor_id(), &first.actor_id().child_id(1));
        assert_eq!(third.actor_id(), &first.actor_id().child_id(2));

        // Supervision tree is constructed correctly.
        validate_link(third.cell(), second.cell());
        validate_link(second.cell(), first.cell());
        assert!(first.cell().state.parent.upgrade().is_none());

        // Supervision tree is torn down correctly.
        third.drain_and_stop().unwrap();
        third.await;
        assert!(second.cell().state.children.is_empty());
        validate_link(second.cell(), first.cell());

        second.drain_and_stop().unwrap();
        second.await;
        assert!(first.cell().state.children.is_empty());
    }

    #[tokio::test]
    async fn test_child_lifecycle() {
        let proc = Proc::local();

        let root = proc.spawn::<TestActor>("root", ()).await.unwrap();
        let root_1 = TestActor::spawn_child(&root).await;
        let root_2 = TestActor::spawn_child(&root).await;
        let root_2_1 = TestActor::spawn_child(&root_2).await;

        root.drain_and_stop().unwrap();
        root.await;

        for actor in [root_1, root_2, root_2_1] {
            assert!(actor.send(TestActorMessage::Noop()).is_err());
            assert_matches!(actor.await, ActorStatus::Stopped);
        }
    }

    #[tokio::test]
    async fn test_parent_failure() {
        let proc = Proc::local();
        // Need to set a supervison coordinator for this Proc because there will
        // be actor failure(s) in this test which trigger supervision.
        ProcSupervisionCoordinator::set(&proc).await.unwrap();

        let root = proc.spawn::<TestActor>("root", ()).await.unwrap();
        let root_1 = TestActor::spawn_child(&root).await;
        let root_2 = TestActor::spawn_child(&root).await;
        let root_2_1 = TestActor::spawn_child(&root_2).await;

        root_2
            .send(TestActorMessage::Fail(anyhow::anyhow!(
                "some random failure"
            )))
            .unwrap();
        let root_2_actor_id = root_2.actor_id().clone();
        assert_matches!(
            root_2.await,
            ActorStatus::Failed(err) if err == format!("serving {}: processing error: some random failure", root_2_actor_id)
        );

        // TODO: should we provide finer-grained stop reasons, e.g., to indicate it was
        // stopped by a parent failure?
        assert_matches!(root_2_1.await, ActorStatus::Stopped);

        for actor in [root_1, root] {
            // The other actors were unaffected.
            assert_matches!(*actor.status().borrow(), ActorStatus::Idle);
        }
    }

    #[tokio::test]
    async fn test_actor_ledger() {
        async fn wait_until_idle(actor_handle: &ActorHandle<TestActor>) {
            actor_handle
                .status()
                .wait_for(|state: &ActorStatus| matches!(*state, ActorStatus::Idle))
                .await
                .unwrap();
        }

        let proc = Proc::local();

        // Add the 1st root. This root will remain active until the end of the test.
        let root: ActorHandle<TestActor> = proc.spawn::<TestActor>("root", ()).await.unwrap();
        wait_until_idle(&root).await;
        {
            let snapshot = proc.state().ledger.snapshot();
            assert_eq!(
                snapshot.roots,
                hashmap! {
                    root.actor_id().clone() =>
                        ActorTreeSnapshot::empty_typed(0, "hyperactor::proc::tests::TestActor".to_string())
                },
            );
        }

        // Add the 2nd root.
        let another_root: ActorHandle<TestActor> =
            proc.spawn::<TestActor>("another_root", ()).await.unwrap();
        wait_until_idle(&another_root).await;
        {
            let snapshot = proc.state().ledger.snapshot();
            assert_eq!(
                snapshot.roots,
                hashmap! {
                    root.actor_id().clone() =>
                        ActorTreeSnapshot::empty_typed(0, "hyperactor::proc::tests::TestActor".to_string()),
                    another_root.actor_id().clone() =>
                        ActorTreeSnapshot::empty_typed(0, "hyperactor::proc::tests::TestActor".to_string()),
                },
            );
        }

        // Stop the 2nd root. It should be excluded from the snapshot after it
        // is stopped.
        another_root.drain_and_stop().unwrap();
        another_root.await;
        {
            let snapshot = proc.state().ledger.snapshot();
            assert_eq!(
                snapshot.roots,
                hashmap! { root.actor_id().clone() =>
                    ActorTreeSnapshot::empty_typed(0, "hyperactor::proc::tests::TestActor".to_string())
                },
            );
        }

        // Incrementally add the following children tree to root. This tree
        // should be captured by snapshot.
        //     root -> root_1 -> root_1_1
        //         |-> root_2

        let root_1 = TestActor::spawn_child(&root).await;
        wait_until_idle(&root_1).await;
        {
            let snapshot = proc.state().ledger.snapshot();
            assert_eq!(
                snapshot.roots,
                hashmap! {
                    root.actor_id().clone() =>  ActorTreeSnapshot {
                        pid: 0,
                        type_name: "hyperactor::proc::tests::TestActor".to_string(),
                        status: ActorStatus::Idle,
                        stats: ActorStats { num_processed_messages: 1 },
                        handlers: HashMap::new(),
                        children: hashmap! {
                            root_1.actor_id().pid() =>
                                ActorTreeSnapshot::empty_typed(
                                    root_1.actor_id().pid(),
                                    "hyperactor::proc::tests::TestActor".to_string()
                                )
                        },
                        events: Vec::new(),
                        spans: Vec::new(),
                    }
                },
            );
        }

        let root_1_1 = TestActor::spawn_child(&root_1).await;
        wait_until_idle(&root_1_1).await;
        {
            let snapshot = proc.state().ledger.snapshot();
            assert_eq!(
                snapshot.roots,
                hashmap! {
                    root.actor_id().clone() =>  ActorTreeSnapshot {
                        pid: 0,
                        type_name: "hyperactor::proc::tests::TestActor".to_string(),
                        status: ActorStatus::Idle,
                        stats: ActorStats { num_processed_messages: 1 },
                        handlers: HashMap::new(),
                        children: hashmap!{
                            root_1.actor_id().pid() =>
                                ActorTreeSnapshot {
                                    pid: root_1.actor_id().pid(),
                                    type_name: "hyperactor::proc::tests::TestActor".to_string(),
                                    status: ActorStatus::Idle,
                                    stats: ActorStats { num_processed_messages: 1 },
                                    handlers: HashMap::new(),
                                    children: hashmap!{
                                        root_1_1.actor_id().pid() =>
                                            ActorTreeSnapshot::empty_typed(
                                                root_1_1.actor_id().pid(),
                                                "hyperactor::proc::tests::TestActor".to_string()
                                            )
                                    },
                                    events: Vec::new(),
                                    spans: Vec::new(),
                                }
                        },
                        events: Vec::new(),
                        spans: Vec::new(),
                    },
                }
            );
        }

        let root_2 = TestActor::spawn_child(&root).await;
        wait_until_idle(&root_2).await;
        {
            let snapshot = proc.state().ledger.snapshot();
            assert_eq!(
                snapshot.roots,
                hashmap! {
                    root.actor_id().clone() =>  ActorTreeSnapshot {
                        pid: 0,
                        type_name: "hyperactor::proc::tests::TestActor".to_string(),
                        status: ActorStatus::Idle,
                        stats: ActorStats { num_processed_messages: 2 },
                        handlers: HashMap::new(),
                        children: hashmap!{
                            root_2.actor_id().pid() =>
                                ActorTreeSnapshot{
                                    pid: root_2.actor_id().pid(),
                                    type_name: "hyperactor::proc::tests::TestActor".to_string(),
                                    status: ActorStatus::Idle,
                                    stats: ActorStats::default(),
                                    handlers: HashMap::new(),
                                    children: HashMap::new(),
                                    events: Vec::new(),
                                    spans: Vec::new(),
                                },
                            root_1.actor_id().pid() =>
                                ActorTreeSnapshot{
                                    pid: root_1.actor_id().pid(),
                                    type_name: "hyperactor::proc::tests::TestActor".to_string(),
                                    status: ActorStatus::Idle,
                                    stats: ActorStats { num_processed_messages: 1 },
                                    handlers: HashMap::new(),
                                    children: hashmap!{
                                        root_1_1.actor_id().pid() =>
                                            ActorTreeSnapshot::empty_typed(
                                                root_1_1.actor_id().pid(),
                                                "hyperactor::proc::tests::TestActor".to_string()
                                            )
                                    },
                                    events: Vec::new(),
                                    spans: Vec::new(),
                                },
                        },
                        events: Vec::new(),
                        spans: Vec::new(),
                    },
                }
            );
        }

        // Stop root_1. This should remove it, and its child, from snapshot.
        root_1.drain_and_stop().unwrap();
        root_1.await;
        {
            let snapshot = proc.state().ledger.snapshot();
            assert_eq!(
                snapshot.roots,
                hashmap! {
                    root.actor_id().clone() =>  ActorTreeSnapshot {
                        pid: 0,
                        type_name: "hyperactor::proc::tests::TestActor".to_string(),
                        status: ActorStatus::Idle,
                        stats: ActorStats { num_processed_messages: 3 },
                        handlers: HashMap::new(),
                        children: hashmap!{
                            root_2.actor_id().pid() =>
                                ActorTreeSnapshot {
                                    pid: root_2.actor_id().pid(),
                                    type_name: "hyperactor::proc::tests::TestActor".to_string(),
                                    status: ActorStatus::Idle,
                                    stats: ActorStats::default(),
                                    handlers: HashMap::new(),
                                    children: HashMap::new(),
                                    events: Vec::new(),
                                    spans: Vec::new(),
                                }
                        },
                        events: Vec::new(),
                        spans: Vec::new(),
                    },
                }
            );
        }

        // Finally stop root. No roots should be left in snapshot.
        root.drain_and_stop().unwrap();
        root.await;
        {
            let snapshot = proc.state().ledger.snapshot();
            assert_eq!(snapshot.roots, hashmap! {});
        }
    }

    #[tokio::test]
    async fn test_multi_handler() {
        // TEMPORARY: This test is currently a bit awkward since we don't yet expose
        // public interfaces to multi-handlers. This will be fixed shortly.

        #[derive(Debug)]
        struct TestActor(Arc<AtomicUsize>);

        #[async_trait]
        impl Actor for TestActor {
            type Params = Arc<AtomicUsize>;

            async fn new(param: Arc<AtomicUsize>) -> Result<Self, anyhow::Error> {
                Ok(Self(param))
            }
        }

        #[async_trait]
        impl Handler<OncePortHandle<PortHandle<usize>>> for TestActor {
            async fn handle(
                &mut self,
                this: &Instance<Self>,
                message: OncePortHandle<PortHandle<usize>>,
            ) -> anyhow::Result<()> {
                message.send(this.port())?;
                Ok(())
            }
        }

        #[async_trait]
        impl Handler<usize> for TestActor {
            async fn handle(
                &mut self,
                _this: &Instance<Self>,
                message: usize,
            ) -> anyhow::Result<()> {
                self.0.fetch_add(message, Ordering::SeqCst);
                Ok(())
            }
        }

        let proc = Proc::local();
        let state = Arc::new(AtomicUsize::new(0));
        let handle = proc
            .spawn::<TestActor>("test", state.clone())
            .await
            .unwrap();
        let client = proc.attach("client").unwrap();
        let (tx, rx) = client.open_once_port();
        handle.send(tx).unwrap();
        let usize_handle = rx.recv().await.unwrap();
        usize_handle.send(123).unwrap();

        handle.drain_and_stop().unwrap();
        handle.await;

        assert_eq!(state.load(Ordering::SeqCst), 123);
    }

    #[tokio::test]
    async fn test_actor_panic() {
        // Need this custom hook to store panic backtrace in task_local.
        panic_handler::set_panic_hook();

        let proc = Proc::local();
        // Need to set a supervison coordinator for this Proc because there will
        // be actor failure(s) in this test which trigger supervision.
        ProcSupervisionCoordinator::set(&proc).await.unwrap();

        let client = proc.attach("client").unwrap();
        let actor_handle = proc.spawn::<TestActor>("test", ()).await.unwrap();
        actor_handle
            .panic(&client, "some random failure".to_string())
            .await
            .unwrap();
        let actor_status = actor_handle.await;

        // Note: even when the test passes, the panic stacktrace will still be
        // printed to stderr because that is the behavior controlled by the panic
        // hook.
        assert_matches!(actor_status, ActorStatus::Failed(_));
        if let ActorStatus::Failed(err) = actor_status {
            let error_msg = err.to_string();
            // Verify panic message is captured
            assert!(error_msg.contains("some random failure"));
            // Verify backtrace is captured. Note the backtrace message might
            // change in the future. If that happens, we need to update this
            // statement with something up-to-date.
            assert!(error_msg.contains("rust_begin_unwind"));
        }
    }

    #[tokio::test]
    async fn test_local_supervision_propagation() {
        #[derive(Debug)]
        struct TestActor(Arc<AtomicBool>, bool);

        #[async_trait]
        impl Actor for TestActor {
            type Params = (Arc<AtomicBool>, bool);

            async fn new(param: (Arc<AtomicBool>, bool)) -> Result<Self, anyhow::Error> {
                Ok(Self(param.0, param.1))
            }

            async fn handle_supervision_event(
                &mut self,
                _this: &Instance<Self>,
                _event: &ActorSupervisionEvent,
            ) -> Result<bool, anyhow::Error> {
                if !self.1 {
                    return Ok(false);
                }

                tracing::error!(
                    "{}: supervision event received: {:?}",
                    _this.self_id(),
                    _event
                );
                self.0.store(true, Ordering::SeqCst);
                Ok(true)
            }
        }

        #[async_trait]
        impl Handler<String> for TestActor {
            async fn handle(
                &mut self,
                this: &Instance<Self>,
                message: String,
            ) -> anyhow::Result<()> {
                tracing::info!("{} received message: {}", this.self_id(), message);
                Err(anyhow::anyhow!(message))
            }
        }

        let proc = Proc::local();
        let reported_event = ProcSupervisionCoordinator::set(&proc).await.unwrap();

        let root_state = Arc::new(AtomicBool::new(false));
        let root_1_state = Arc::new(AtomicBool::new(false));
        let root_1_1_state = Arc::new(AtomicBool::new(false));
        let root_1_1_1_state = Arc::new(AtomicBool::new(false));
        let root_2_state = Arc::new(AtomicBool::new(false));
        let root_2_1_state = Arc::new(AtomicBool::new(false));

        let root = proc
            .spawn::<TestActor>("root", (root_state.clone(), false))
            .await
            .unwrap();
        let root_1 = proc
            .spawn_child::<TestActor>(
                root.cell().clone(),
                (
                    root_1_state.clone(),
                    true, /* set true so children's event stops here */
                ),
            )
            .await
            .unwrap();
        let root_1_1 = proc
            .spawn_child::<TestActor>(root_1.cell().clone(), (root_1_1_state.clone(), false))
            .await
            .unwrap();
        let root_1_1_1 = proc
            .spawn_child::<TestActor>(root_1_1.cell().clone(), (root_1_1_1_state.clone(), false))
            .await
            .unwrap();
        let root_2 = proc
            .spawn_child::<TestActor>(root.cell().clone(), (root_2_state.clone(), false))
            .await
            .unwrap();
        let root_2_1 = proc
            .spawn_child::<TestActor>(root_2.cell().clone(), (root_2_1_state.clone(), false))
            .await
            .unwrap();

        // fail `root_1_1_1`, the supervision msg should be propagated to
        // `root_1` because `root_1` has set `true` to `handle_supervision_event`.
        root_1_1_1
            .send::<String>("some random failure".into())
            .unwrap();

        // fail `root_2_1`, the supervision msg should be propagated to
        // ProcSupervisionCoordinator.
        root_2_1
            .send::<String>("some random failure".into())
            .unwrap();

        RealClock.sleep(Duration::from_secs(1)).await;

        assert!(!root_state.load(Ordering::SeqCst));
        assert!(root_1_state.load(Ordering::SeqCst));
        assert!(!root_1_1_state.load(Ordering::SeqCst));
        assert!(!root_1_1_1_state.load(Ordering::SeqCst));
        assert!(!root_2_state.load(Ordering::SeqCst));
        assert!(!root_2_1_state.load(Ordering::SeqCst));
        assert_eq!(
            reported_event.event().map(|e| e.actor_id().clone()),
            Some(root_2_1.actor_id().clone())
        );
    }

    #[tokio::test]
    async fn test_proc_terminate_without_coordinator() {
        if std::env::var("CARGO_TEST").is_ok() {
            eprintln!("test skipped as it hangs when run by cargo in sandcastle");
            return;
        }

        let process = async {
            let proc = Proc::local();
            // Intentionally not setting a proc supervison coordinator. This
            // should cause the process to terminate.
            // ProcSupervisionCoordinator::set(&proc).await.unwrap();
            let root = proc.spawn::<TestActor>("root", ()).await.unwrap();
            let client = proc.attach("client").unwrap();
            root.fail(&client, anyhow::anyhow!("some random failure"))
                .await
                .unwrap();
            // It is okay to sleep a long time here, because we expect this
            // process to be terminated way before the sleep ends due to the
            // missing proc supervison coordinator.
            RealClock.sleep(Duration::from_secs(30)).await;
        };

        assert_termination(|| process, 1).await.unwrap();
    }

    fn trace_and_block(fut: impl Future) {
        tracing::subscriber::with_default(
            tracing_subscriber::Registry::default().with(hyperactor_telemetry::recorder().layer()),
            || {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(fut)
            },
        );
    }

    #[ignore = "until trace recording is turned back on"]
    #[test]
    fn test_handler_logging() {
        #[derive(Debug)]
        struct LoggingActor;

        impl LoggingActor {
            async fn wait(handle: &ActorHandle<Self>) {
                let barrier = Arc::new(Barrier::new(2));
                handle.send(barrier.clone()).unwrap();
                barrier.wait().await;
            }
        }

        #[async_trait]
        impl Actor for LoggingActor {
            type Params = ();

            async fn new(_params: ()) -> Result<Self, anyhow::Error> {
                Ok(Self)
            }
        }

        #[async_trait]
        impl Handler<String> for LoggingActor {
            async fn handle(
                &mut self,
                _this: &Instance<Self>,
                message: String,
            ) -> anyhow::Result<()> {
                tracing::info!("{}", message);
                Ok(())
            }
        }

        #[async_trait]
        impl Handler<u64> for LoggingActor {
            async fn handle(&mut self, _this: &Instance<Self>, message: u64) -> anyhow::Result<()> {
                tracing::event!(Level::INFO, number = message);
                Ok(())
            }
        }

        #[async_trait]
        impl Handler<Arc<Barrier>> for LoggingActor {
            async fn handle(
                &mut self,
                _this: &Instance<Self>,
                message: Arc<Barrier>,
            ) -> anyhow::Result<()> {
                message.wait().await;
                Ok(())
            }
        }

        #[async_trait]
        impl Handler<Arc<(Barrier, Barrier)>> for LoggingActor {
            async fn handle(
                &mut self,
                _this: &Instance<Self>,
                barriers: Arc<(Barrier, Barrier)>,
            ) -> anyhow::Result<()> {
                let inner = tracing::span!(Level::INFO, "child_span");
                let _inner_guard = inner.enter();
                barriers.0.wait().await;
                barriers.1.wait().await;
                Ok(())
            }
        }

        trace_and_block(async {
            let handle = LoggingActor::spawn_detached(()).await.unwrap();
            handle.send("hello world".to_string()).unwrap();
            handle.send("hello world again".to_string()).unwrap();
            handle.send(123u64).unwrap();

            LoggingActor::wait(&handle).await;

            let events = handle.cell().state.recording.tail();
            assert_eq!(events.len(), 3);
            assert_eq!(events[0].json_value(), json!({ "message": "hello world" }));
            assert_eq!(
                events[1].json_value(),
                json!({ "message": "hello world again" })
            );
            assert_eq!(events[2].json_value(), json!({ "number": 123 }));

            let stacks = {
                let barriers = Arc::new((Barrier::new(2), Barrier::new(2)));
                handle.send(Arc::clone(&barriers)).unwrap();
                barriers.0.wait().await;
                let stacks = handle.cell().state.recording.stacks();
                barriers.1.wait().await;
                stacks
            };
            assert_eq!(stacks.len(), 1);
            assert_eq!(stacks[0].len(), 1);
            assert_eq!(stacks[0][0].name(), "child_span");
        })
    }
}
