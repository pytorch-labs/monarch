//! Mailboxes are the central message-passing mechanism in Hyperactor.
//!
//! Each actor owns a mailbox to which other actors can deliver messages.
//! An actor can open one or more typed _ports_ in the mailbox; messages
//! are in turn delivered to specific ports.
//!
//! Mailboxes are associated with an [`ActorId`] (given by `actor_id`
//! in the following example):
//!
//! ```
//! # use hyperactor::mailbox::Mailbox;
//! # use hyperactor::reference::{ActorId, ProcId, WorldId};
//! # tokio_test::block_on(async {
//! # let proc_id = ProcId(WorldId("world".to_string()), 0);
//! # let actor_id = ActorId(proc_id, "actor".to_string(), 0);
//! let mbox = Mailbox::new_detached(actor_id);
//! let (port, mut receiver) = mbox.open_port::<u64>();
//!
//! port.send(123).unwrap();
//! assert_eq!(receiver.recv().await.unwrap(), 123u64);
//! # })
//! ```
//!
//! Mailboxes also provide a form of one-shot ports, called [`OncePort`],
//! that permits at most one message transmission:
//!
//! ```
//! # use hyperactor::mailbox::Mailbox;
//! # use hyperactor::reference::{ActorId, ProcId, WorldId};
//! # tokio_test::block_on(async {
//! # let proc_id = ProcId(WorldId("world".to_string()), 0);
//! # let actor_id = ActorId(proc_id, "actor".to_string(), 0);
//! let mbox = Mailbox::new_detached(actor_id);
//!
//! let (port, receiver) = mbox.open_once_port::<u64>();
//!
//! port.send(123u64).unwrap();
//! assert_eq!(receiver.recv().await.unwrap(), 123u64);
//! # })
//! ```
//!
//! [`OncePort`]s are correspondingly used for RPC replies in the actor
//! system.
//!
//! ## Remote ports and serialization
//!
//! Mailboxes allow delivery of serialized messages to named ports:
//!
//! 1) Ports restrict message types to (serializable) [`Message`]s.
//! 2) Each [`Port`] is associated with a [`PortId`] which globally names the port.
//! 3) [`Mailbox`] provides interfaces to deliver serialized
//!    messages to ports named by their [`PortId`].
//!
//! While this complicates the interface somewhat, it allows the
//! implementation to avoid a serialization roundtrip when passing
//! messages locally.

#![allow(dead_code)] // Allow until this is used outside of tests.

use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::fmt::Debug;
use std::future::Future;
use std::ops::Bound::Excluded;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::sync::Weak;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use async_trait::async_trait;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate as hyperactor; // for macros
use crate::Named;
use crate::OncePortRef;
use crate::PortRef;
use crate::accum::Accumulator;
use crate::actor::Signal;
use crate::actor::remote::USER_PORT_OFFSET;
use crate::cap;
use crate::channel;
use crate::channel::ChannelAddr;
use crate::channel::ChannelError;
use crate::channel::SendError;
use crate::channel::TxStatus;
use crate::data::Serialized;
use crate::id;
use crate::reference::ActorId;
use crate::reference::PortId;
use crate::reference::Reference;

/// For [`MailboxAdminMessage`], a message type for mailbox administration.
pub mod mailbox_admin_message;
pub use mailbox_admin_message::MailboxAdminMessage;
pub use mailbox_admin_message::MailboxAdminMessageHandler;
/// For [`DurableMailboxSender`] a sender with a write-ahead log.
pub mod durable_mailbox_sender;
pub use durable_mailbox_sender::log;
use durable_mailbox_sender::log::*;

// Port handle and receiver for undeliverable messages.
fn new_undeliverable_port() -> (
    PortHandle<Undeliverable<MessageEnvelope>>,
    PortReceiver<Undeliverable<MessageEnvelope>>,
) {
    Mailbox::new_detached(hyperactor::id!(world[0].proc))
        .open_port::<Undeliverable<MessageEnvelope>>()
}
// An undeliverable message port handle to be shared amongst multiple
// producers. Messages sent here are forwarded to the undeliverable
// mailbox sender.
static MONITORED_RETURN_HANDLE: OnceLock<PortHandle<Undeliverable<MessageEnvelope>>> =
    OnceLock::new();
/// Accessor to the shared monitored undeliverable message port
/// handle. Initialization spawns the undeliverable message port
/// monitor that forwards incoming messages to the undeliverable
/// mailbox sender.
pub fn monitored_return_handle() -> PortHandle<Undeliverable<MessageEnvelope>> {
    let return_handle = MONITORED_RETURN_HANDLE.get_or_init(|| {
        let (return_handle, mut rx) = new_undeliverable_port();
        // Don't reuse `return_handle` for `h`: else it will never get
        // dropped and the task will never return.
        let (h, _) = new_undeliverable_port();
        super::init::RUNTIME.spawn(async move {
            while let Ok(Undeliverable(mut envelope)) = rx.recv().await {
                envelope.try_set_error(DeliveryError::BrokenLink(
                    "message returned to undeliverable port".to_string(),
                ));
                UndeliverableMailboxSender.post(envelope, /*unused */ h.clone());
            }
        });
        return_handle
    });

    return_handle.clone()
}

/// Returns a message envelope to its original sender.
fn return_undeliverable(
    return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    envelope: MessageEnvelope,
) {
    super::init::RUNTIME.spawn(async move {
        let envelope_copy = envelope.clone();
        if (return_handle.send(Undeliverable(envelope))).is_err() {
            UndeliverableMailboxSender.post(envelope_copy, /*unsued*/ return_handle)
        }
    });
}

/// Message collects the necessary requirements for messages that are deposited
/// into mailboxes.
pub trait Message: Debug + Send + Sync + 'static {}
impl<M: Debug + Send + Sync + 'static> Message for M {}

/// RemoteMessage extends [`Message`] by requiring that the messages
/// also be serializable, and can thus traverse process boundaries.
/// RemoteMessages must also specify a globally unique type name (a URI).
pub trait RemoteMessage: Message + Named + Serialize + DeserializeOwned {}

impl<M: Message + Named + Serialize + DeserializeOwned> RemoteMessage for M {}

/// Type alias for bytestring data used throughout the system.
pub type Data = Vec<u8>;

/// Delivery errors occur during message posting.
#[derive(
    thiserror::Error,
    Debug,
    Serialize,
    Deserialize,
    Named,
    Clone,
    PartialEq
)]
pub enum DeliveryError {
    /// The destination address is not reachable.
    #[error("address not routable: {0}")]
    Unroutable(String),

    /// A broken link indicates that a link in the message
    /// delivery path has failed.
    #[error("broken link: {0}")]
    BrokenLink(String),

    /// A (local) mailbox delivery error.
    #[error("mailbox error: {0}")]
    Mailbox(String),
}

/// An envelope that carries a message destined to a remote actor.
/// The envelope contains a serialized message along with its destination
/// and sender.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Named)]
pub struct MessageEnvelope {
    /// The sender of this message.
    sender: ActorId,

    /// The destination of the message.
    dest: PortId,

    /// The serialized message.
    data: Serialized,

    /// Error contains a delivery error when message delivery failed.
    error: Option<DeliveryError>,
    // TODO: add typename, source, seq, TTL, etc.
}

impl MessageEnvelope {
    /// Create a new envelope with the provided sender, destination, and message.
    pub fn new(sender: ActorId, dest: PortId, data: Serialized) -> Self {
        Self {
            sender,
            dest,
            data,
            error: None,
        }
    }

    /// Create a new envelope whose sender ID is unknown.
    pub(crate) fn new_unknown(dest: PortId, data: Serialized) -> Self {
        Self::new(id!(unknown[0].unknown), dest, data)
    }

    /// Construct a new serialized value by serializing the provided T-typed value.
    pub fn serialize<T: Serialize + Named>(
        source: ActorId,
        dest: PortId,
        value: &T,
    ) -> Result<Self, bincode::Error> {
        Ok(Self {
            data: Serialized::serialize(value)?,
            sender: source,
            dest,
            error: None,
        })
    }

    /// Deserialize the message in the envelope to the provided type T.
    pub fn deserialized<T: DeserializeOwned>(&self) -> Result<T, anyhow::Error> {
        self.data.deserialized()
    }

    /// The serialized message.
    pub fn data(&self) -> &Serialized {
        &self.data
    }

    /// The message sender.
    pub fn sender(&self) -> &ActorId {
        &self.sender
    }

    /// The destination of the message.
    pub fn dest(&self) -> &PortId {
        &self.dest
    }

    /// Tells whether this is a signal message.
    pub fn is_signal(&self) -> bool {
        self.dest.index() == Signal::port()
    }

    /// Tries to sets a delivery error for the message. If the error is already
    /// set, nothing is updated.
    fn try_set_error(&mut self, error: DeliveryError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    /// The message has been determined to be undeliverable with the
    /// provided error. Mark the envelope with the error and return to
    /// sender.
    pub fn undeliverable(
        mut self,
        error: DeliveryError,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        self.try_set_error(error);
        return_undeliverable(return_handle, self);
    }

    /// Get the error of why this message was undeliverable. None means this
    /// message was not determined as undeliverable.
    pub fn error(&self) -> Option<&DeliveryError> {
        self.error.as_ref()
    }

    fn open(self) -> (MessageMetadata, Serialized) {
        let Self {
            sender,
            dest,
            data,
            error,
        } = self;

        (
            MessageMetadata {
                sender,
                dest,
                error,
            },
            data,
        )
    }

    fn seal(metadata: MessageMetadata, data: Serialized) -> Self {
        let MessageMetadata {
            sender,
            dest,
            error,
        } = metadata;

        Self {
            sender,
            dest,
            data,
            error,
        }
    }
}

impl fmt::Display for MessageEnvelope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.error {
            None => write!(f, "{} > {}: {}", self.sender, self.dest, self.data),
            Some(err) => write!(
                f,
                "{} > {}: {}: delivery error: {}",
                self.sender, self.dest, self.data, err
            ),
        }
    }
}

struct MessageMetadata {
    sender: ActorId,
    dest: PortId,
    error: Option<DeliveryError>,
}

/// Errors that occur during mailbox operations. Each error is associated
/// with the mailbox's actor id.
#[derive(Debug)]
pub struct MailboxError {
    actor_id: ActorId,
    kind: MailboxErrorKind,
}

/// The kinds of mailbox errors. This enum is marked non-exhaustive to
/// allow for extensibility.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum MailboxErrorKind {
    /// An operation was attempted on a closed mailbox.
    #[error("mailbox closed")]
    Closed,

    /// The port associated with an operation was invalid.
    #[error("invalid port: {0}")]
    InvalidPort(PortId),

    /// There was no sender associated with the port.
    #[error("no sender for port: {0}")]
    NoSenderForPort(PortId),

    /// There was no local sender associated with the port.
    /// Returned by operations that require a local port.
    #[error("no local sender for port: {0}")]
    NoLocalSenderForPort(PortId),

    /// The port was closed.
    #[error("{0}: port closed")]
    PortClosed(PortId),

    /// An error occured during a send operation.
    #[error("send {0}: {1}")]
    Send(PortId, #[source] anyhow::Error),

    /// An error occured during a receive operation.
    #[error("recv {0}: {1}")]
    Recv(PortId, #[source] anyhow::Error),

    /// There was a serialization failure.
    #[error("serialize: {0}")]
    Serialize(#[source] anyhow::Error),

    /// There was a deserialization failure.
    #[error("deserialize {0}: {1}")]
    Deserialize(&'static str, anyhow::Error),

    /// There was an error during a channel operation.
    #[error(transparent)]
    Channel(#[from] ChannelError),
}

impl MailboxError {
    /// Create a new mailbox error associated with the provided actor
    /// id and of the given kind.
    pub fn new(actor_id: ActorId, kind: MailboxErrorKind) -> Self {
        Self { actor_id, kind }
    }

    /// The ID of the mailbox producing this error.
    pub fn actor_id(&self) -> &ActorId {
        &self.actor_id
    }

    /// The error's kind.
    pub fn kind(&self) -> &MailboxErrorKind {
        &self.kind
    }
}

impl fmt::Display for MailboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.actor_id)?;
        fmt::Display::fmt(&self.kind, f)
    }
}

impl std::error::Error for MailboxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.kind.source()
    }
}

/// PortLocation describes the location of a port.
/// This is used in errors to provide a uniform data type
/// for ports that may or may not be bound.
#[derive(Debug, Clone)]
pub enum PortLocation {
    /// The port was bound: the location is its underlying bound ID.
    Bound(PortId),
    /// The port was not bound: we provide the actor ID and the message type.
    Unbound(ActorId, &'static str),
}

impl PortLocation {
    fn new_unbound<M: Message>(actor_id: ActorId) -> Self {
        PortLocation::Unbound(actor_id, std::any::type_name::<M>())
    }

    fn new_unbound_type(actor_id: ActorId, ty: &'static str) -> Self {
        PortLocation::Unbound(actor_id, ty)
    }

    /// The actor id of the location.
    pub fn actor_id(&self) -> &ActorId {
        match self {
            PortLocation::Bound(port_id) => port_id.actor_id(),
            PortLocation::Unbound(actor_id, _) => actor_id,
        }
    }
}

impl fmt::Display for PortLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PortLocation::Bound(port_id) => write!(f, "{}", port_id),
            PortLocation::Unbound(actor_id, name) => write!(f, "{}<{}>", actor_id, name),
        }
    }
}

/// Errors that that occur during mailbox sending operations. Each error
/// is associated with the port ID of the operation.
#[derive(Debug)]
pub struct MailboxSenderError {
    location: PortLocation,
    kind: MailboxSenderErrorKind,
}

/// The kind of mailbox sending errors.
#[derive(thiserror::Error, Debug)]
pub enum MailboxSenderErrorKind {
    /// Error during serialization.
    #[error("serialization error: {0}")]
    Serialize(anyhow::Error),

    /// Error during deserialization.
    #[error("deserialization error for type {0}: {1}")]
    Deserialize(&'static str, anyhow::Error),

    /// A send to an invalid port.
    #[error("invalid port")]
    Invalid,

    /// A send to a closed port.
    #[error("port closed")]
    Closed,

    // The following pass through underlying errors:
    /// An underlying mailbox error.
    #[error(transparent)]
    Mailbox(#[from] MailboxError),

    /// An underlying channel error.
    #[error(transparent)]
    Channel(#[from] ChannelError),

    /// An underlying message log error.
    #[error(transparent)]
    MessageLog(#[from] MessageLogError),

    /// An other, uncategorized error.
    #[error("send error: {0}")]
    Other(#[from] anyhow::Error),

    /// The destination was unreachable.
    #[error("unreachable: {0}")]
    Unreachable(anyhow::Error),
}

impl MailboxSenderError {
    /// Create a new mailbox sender error to an unbound port.
    pub fn new_unbound<M>(actor_id: ActorId, kind: MailboxSenderErrorKind) -> Self {
        Self {
            location: PortLocation::Unbound(actor_id, std::any::type_name::<M>()),
            kind,
        }
    }

    /// Create a new mailbox sender, manually providing the type.
    pub fn new_unbound_type(
        actor_id: ActorId,
        kind: MailboxSenderErrorKind,
        ty: &'static str,
    ) -> Self {
        Self {
            location: PortLocation::Unbound(actor_id, ty),
            kind,
        }
    }

    /// Create a new mailbox sender error with the provided port ID and kind.
    pub fn new_bound(port_id: PortId, kind: MailboxSenderErrorKind) -> Self {
        Self {
            location: PortLocation::Bound(port_id),
            kind,
        }
    }

    /// The location at which the error occured.
    pub fn location(&self) -> &PortLocation {
        &self.location
    }

    /// The kind associated with the error.
    pub fn kind(&self) -> &MailboxSenderErrorKind {
        &self.kind
    }
}

impl fmt::Display for MailboxSenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.location)?;
        fmt::Display::fmt(&self.kind, f)
    }
}

impl std::error::Error for MailboxSenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.kind.source()
    }
}

/// An undeliverable `M`-typed message (in practice `M` is
/// [MessageEnvelope]).
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Undeliverable<M: Message>(pub M);

/// For `M` a [RemoteMessage], `Undeliverable<M>` is a [Named]
/// instance.
impl<M: RemoteMessage> crate::data::Named for Undeliverable<M> {
    fn typename() -> &'static str {
        crate::data::intern_typename!(Self, "hyperactor::Undeliverable<{}>", M)
    }
}

/// MailboxSenders can send messages through ports to mailboxes. It
/// provides a unified interface for message delivery in the system.
pub trait MailboxSender: Send + Sync + Debug + Any {
    /// TODO: consider making this publicly inaccessible. While the trait itself
    /// needs to be public, its only purpose for the end-user API is to provide
    /// the typed messaging APIs from (Once)PortRef and ActorRef.
    fn post(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    );
}

// PortSender is an extension trait so that we can include generics,
// making the API end-to-end typesafe.

/// PortSender extends [`MailboxSender`] by providing typed endpoints
/// for sending messages over ports.
pub trait PortSender: MailboxSender {
    /// Deliver a message to the provided port.
    fn serialize_and_send<M: RemoteMessage>(
        &self,
        port: &PortRef<M>,
        message: M,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) -> Result<(), MailboxSenderError> {
        // TODO: convert this to a undeliverable error also
        let serialized = Serialized::serialize(&message).map_err(|err| {
            MailboxSenderError::new_bound(
                port.port_id().clone(),
                MailboxSenderErrorKind::Serialize(err.into()),
            )
        })?;
        self.post(
            MessageEnvelope::new_unknown(port.port_id().clone(), serialized),
            return_handle,
        );
        Ok(())
    }

    /// Deliver a message to a one-shot port, consuming the provided port,
    /// which is not reusable.
    fn serialize_and_send_once<M: RemoteMessage>(
        &self,
        once_port: OncePortRef<M>,
        message: M,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) -> Result<(), MailboxSenderError> {
        let serialized = Serialized::serialize(&message).map_err(|err| {
            MailboxSenderError::new_bound(
                once_port.port_id().clone(),
                MailboxSenderErrorKind::Serialize(err.into()),
            )
        })?;
        self.post(
            MessageEnvelope::new_unknown(once_port.port_id().clone(), serialized),
            return_handle,
        );
        Ok(())
    }
}

impl<T: ?Sized + MailboxSender> PortSender for T {}

/// A perpetually closed mailbox sender. Panics if any messages are posted.
/// Useful for tests, or where there is no meaningful mailbox sender
/// implementation available.
#[derive(Debug, Clone)]
pub struct PanickingMailboxSender;

impl MailboxSender for PanickingMailboxSender {
    fn post(
        &self,
        envelope: MessageEnvelope,
        _return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        panic!("panic! in the mailbox! attempted post: {}", envelope)
    }
}

/// A mailbox sender for undeliverable messages. This will simply record
/// any undelivered messages.
#[derive(Debug)]
pub struct UndeliverableMailboxSender;

impl MailboxSender for UndeliverableMailboxSender {
    fn post(
        &self,
        envelope: MessageEnvelope,
        _return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        tracing::error!("message not delivered: {}", envelope);
    }
}

#[derive(Debug)]
struct Buffer<T: Message> {
    queue: mpsc::UnboundedSender<(T, PortHandle<Undeliverable<T>>)>,
    processed: watch::Receiver<usize>,
    seq: AtomicUsize,
}

impl<T: Message> Buffer<T> {
    fn new<Fut>(
        process: impl Fn(T, PortHandle<Undeliverable<T>>) -> Fut + Send + Sync + 'static,
    ) -> Self
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        let (queue, mut next) = mpsc::unbounded_channel();
        let (last_processed, processed) = watch::channel(0);
        crate::init::RUNTIME.spawn(async move {
            let mut seq = 0;
            while let Some((msg, return_handle)) = next.recv().await {
                process(msg, return_handle).await;
                seq += 1;
                let _ = last_processed.send(seq);
            }
        });
        Self {
            queue,
            processed,
            seq: AtomicUsize::new(0),
        }
    }

    fn send(
        &self,
        item: (T, PortHandle<Undeliverable<T>>),
    ) -> Result<(), mpsc::error::SendError<(T, PortHandle<Undeliverable<T>>)>> {
        self.seq.fetch_add(1, Ordering::SeqCst);
        self.queue.send(item)?;
        Ok(())
    }

    async fn flush(&mut self) -> Result<(), watch::error::RecvError> {
        let seq = self.seq.load(Ordering::SeqCst);
        while *self.processed.borrow_and_update() < seq {
            self.processed.changed().await?;
        }
        Ok(())
    }
}

static BOXED_PANICKING_MAILBOX_SENDER: LazyLock<BoxedMailboxSender> =
    LazyLock::new(|| BoxedMailboxSender::new(PanickingMailboxSender));

/// Convenience boxing implementation for MailboxSender. Most APIs
/// are parameterized on MailboxSender implementations, and it's thus
/// difficult to work with dyn values.  BoxedMailboxSender bridges this
/// gap by providing a concrete MailboxSender which dispatches using an
/// underlying (boxed) dyn.
#[derive(Debug, Clone)]
pub struct BoxedMailboxSender(Arc<dyn MailboxSender + Send + Sync + 'static>);

impl BoxedMailboxSender {
    /// Create a new boxed sender given the provided sender implementation.
    pub fn new(sender: impl MailboxSender + 'static) -> Self {
        Self(Arc::new(sender))
    }

    /// Attempts to downcast the inner sender to the given concrete
    /// type.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        (&*self.0 as &dyn Any).downcast_ref::<T>()
    }
}

/// Extension trait that creates a boxed clone of a MailboxSender.
pub trait BoxableMailboxSender: MailboxSender + Clone + 'static {
    /// A boxed clone of this MailboxSender.
    fn boxed(&self) -> BoxedMailboxSender;
}
impl<T: MailboxSender + Clone + 'static> BoxableMailboxSender for T {
    fn boxed(&self) -> BoxedMailboxSender {
        BoxedMailboxSender::new(self.clone())
    }
}

/// Extension trait that rehomes a MailboxSender into a BoxedMailboxSender.
pub trait IntoBoxedMailboxSender: MailboxSender {
    /// Rehome this MailboxSender into a BoxedMailboxSender.
    fn into_boxed(self) -> BoxedMailboxSender;
}
impl<T: MailboxSender + 'static> IntoBoxedMailboxSender for T {
    fn into_boxed(self) -> BoxedMailboxSender {
        BoxedMailboxSender::new(self)
    }
}

impl MailboxSender for BoxedMailboxSender {
    fn post(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        hyperactor_telemetry::declare_static_counter!(MAILBOX_POSTS, "mailbox.posts");
        MAILBOX_POSTS.add(
            1,
            hyperactor_telemetry::kv_pairs!(
                "actor_id" => envelope.sender.to_string(),
                "dest_actor_id" => envelope.dest.0.to_string(),
            ),
        );
        self.0.post(envelope, return_handle)
    }
}

/// Errors that occur during mailbox serving.
#[derive(thiserror::Error, Debug)]
pub enum MailboxServerError {
    /// An underlying channel error.
    #[error(transparent)]
    Channel(#[from] ChannelError),

    /// An underlying mailbox sender error.
    #[error(transparent)]
    MailboxSender(#[from] MailboxSenderError),
}

/// Represents a running [`MailboxServer`]. The handle composes a
/// ['tokio::task::JoinHandle'] and may be joined in the same manner.
#[derive(Debug)]
pub struct MailboxServerHandle {
    join_handle: JoinHandle<Result<(), MailboxServerError>>,
    stopped_tx: watch::Sender<bool>,
}

impl MailboxServerHandle {
    /// Signal the server to stop serving the mailbox. The caller should
    /// join the handle by awaiting the [`MailboxServerHandle`] future.
    ///
    /// Stop should be called at most once.
    pub fn stop(&self) {
        self.stopped_tx.send(true).expect("stop called twice");
    }
}

/// Forward future implementation to underlying handle.
impl Future for MailboxServerHandle {
    type Output = <JoinHandle<Result<(), MailboxServerError>> as Future>::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: This is safe to do because self is pinned.
        let join_handle_pinned =
            unsafe { self.map_unchecked_mut(|container| &mut container.join_handle) };
        join_handle_pinned.poll(cx)
    }
}

/// Serve a port on the provided [`channel::Rx`]. This dispatches all
/// channel messages directly to the port.
pub trait MailboxServer: MailboxSender + Sized + 'static {
    /// Serve the provided port on the given channel on this sender on
    /// a background task which may be joined with the returned handle.
    /// The task fails on any send error.
    fn serve(
        self,
        mut rx: impl channel::Rx<MessageEnvelope> + Send + 'static,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) -> MailboxServerHandle {
        let (stopped_tx, mut stopped_rx) = watch::channel(false);
        let join_handle = tokio::spawn(async move {
            let mut detached = false;

            loop {
                if *stopped_rx.borrow_and_update() {
                    break Ok(());
                }

                tokio::select! {
                    message = rx.recv() => {
                        match message {
                            // Relay the message to the port directly.
                            Ok(envelope) => self.post(envelope, return_handle.clone()),

                            // Closed is a "graceful" error in this case.
                            // We simply stop serving.
                            Err(ChannelError::Closed) => break Ok(()),
                            Err(channel_err) => break Err(MailboxServerError::from(channel_err)),
                        }
                    }
                    result = stopped_rx.changed(), if !detached  => {
                        tracing::debug!(
                            "the mailbox server is stopped"
                        );
                        detached = result.is_err();
                    }
                }
            }
        });

        MailboxServerHandle {
            join_handle,
            stopped_tx,
        }
    }
}

impl<T: MailboxSender + Sized + Sync + Send + 'static> MailboxServer for T {}

/// A mailbox server client that transmits messages on a Tx channel.
#[derive(Debug)]
pub struct MailboxClient {
    // The unbounded sender.
    buffer: Buffer<MessageEnvelope>,

    // To cancel monitoring tx health.
    _tx_monitoring: CancellationToken,
}

impl MailboxClient {
    /// Create a new client that sends messages destined for a
    /// [`MailboxServer`] on the provided Tx channel.
    pub fn new(tx: impl channel::Tx<MessageEnvelope> + Send + Sync + 'static) -> Self {
        let addr = tx.addr();
        let tx = Arc::new(tx);
        let tx_status = tx.status().clone();
        let tx_monitoring = CancellationToken::new();
        let buffer = Buffer::new(move |envelope, return_handle| {
            let tx = Arc::clone(&tx);
            let (return_channel, return_receiver) = oneshot::channel();
            // Set up for delivery failure.
            let return_handle_0 = return_handle.clone();
            tokio::spawn(async move {
                let result = return_receiver.await;
                if let Ok(message) = result {
                    let _ = return_handle_0.send(Undeliverable(message));
                } else {
                    // Sender dropped, this task can end.
                }
            });
            // Send the message for transmission.
            let return_handle_1 = return_handle.clone();
            async move {
                if let Err(SendError(_, envelope)) = tx.try_post(envelope, return_channel) {
                    // Failed to enqueue.
                    envelope.undeliverable(
                        DeliveryError::BrokenLink("failed to enqueue in MailboxClient".to_string()),
                        return_handle_1.clone(),
                    );
                }
            }
        });
        let this = Self {
            buffer,
            _tx_monitoring: tx_monitoring.clone(),
        };
        Self::monitor_tx_health(tx_status, tx_monitoring, addr);
        this
    }

    // Set up a watch for the tx's health.
    fn monitor_tx_health(
        mut rx: watch::Receiver<TxStatus>,
        cancel_token: CancellationToken,
        addr: ChannelAddr,
    ) {
        crate::init::RUNTIME.spawn(async move {
            loop {
                tokio::select! {
                    changed = rx.changed() => {
                        if changed.is_err() || *rx.borrow() == TxStatus::Closed {
                            tracing::warn!("connection to {} lost", addr);
                            // TODO: Potential for supervision event
                            // interaction here.
                            break;
                        }
                    }
                    _ = cancel_token.cancelled() => {
                        break;
                    }
                }
            }
        });
    }
}

impl MailboxSender for MailboxClient {
    fn post(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        tracing::trace!(name = "post", "posting message to {}", envelope.dest);
        if let Err(mpsc::error::SendError((envelope, return_handle))) =
            self.buffer.send((envelope, return_handle))
        {
            // Failed to enqueue.
            envelope.undeliverable(
                DeliveryError::BrokenLink("failed to enqueue in MailboxClient".to_string()),
                return_handle,
            );
        }
    }
}

/// A mailbox coordinates message delivery to actors through typed
/// [`Port`]s associated with the mailbox.
#[derive(Clone, Debug)]
pub struct Mailbox {
    state: Arc<State>,
}

impl Mailbox {
    /// Create a new mailbox associated with the provided actor ID, using the provided
    /// forwarder for external destinations.
    pub fn new(actor_id: ActorId, forwarder: BoxedMailboxSender) -> Self {
        Self {
            state: Arc::new(State::new(actor_id, forwarder)),
        }
    }

    /// Create a new detached mailbox associated with the provided actor ID.
    pub fn new_detached(actor_id: ActorId) -> Self {
        Self {
            state: Arc::new(State::new(actor_id, BOXED_PANICKING_MAILBOX_SENDER.clone())),
        }
    }

    /// The actor id associated with this mailbox.
    pub fn actor_id(&self) -> &ActorId {
        &self.state.actor_id
    }

    /// Open a new port that accepts M-typed messages. The returned
    /// port may be freely cloned, serialized, and passed around. The
    /// returned receiver should only be retained by the actor responsible
    /// for processing the delivered messages.
    pub fn open_port<M: Message>(&self) -> (PortHandle<M>, PortReceiver<M>) {
        let port_index = self.state.allocate_port();
        let (sender, receiver) = mpsc::unbounded_channel::<M>();
        let port_id = PortId(self.state.actor_id.clone(), port_index);
        tracing::trace!(
            name = "open_port",
            "opening port for {} at {}",
            self.state.actor_id,
            port_id
        );
        (
            PortHandle::new(self.clone(), port_index, UnboundedPortSender::Mpsc(sender)),
            PortReceiver::new(
                receiver,
                port_id,
                /*coalesce=*/ false,
                self.state.clone(),
            ),
        )
    }

    /// Open a new port with an accumulator. This port accepts A::Update type
    /// messages, accumulate them into A::State with the given accumulator.
    /// The latest changed state can be received from the returned receiver as
    /// a single A::State message. If there is no new update, the receiver will
    /// not receive any message.
    pub fn open_accum_port<A>(&self, accum: A) -> (PortHandle<A::Update>, PortReceiver<A::State>)
    where
        A: Accumulator + Send + Sync + 'static,
        A::Update: Message,
        A::State: Message + Default + Clone,
    {
        let port_index = self.state.allocate_port();
        let (sender, receiver) = mpsc::unbounded_channel::<A::State>();
        let port_id = PortId(self.state.actor_id.clone(), port_index);
        let state = Mutex::new(A::State::default());
        let enqueue = move |update: A::Update| {
            let mut state = state.lock().unwrap();
            accum.accumulate(&mut state, &update);
            let _ = sender.send(state.clone());
            Ok(())
        };
        (
            PortHandle::new(
                self.clone(),
                port_index,
                UnboundedPortSender::Func(Arc::new(enqueue)),
            ),
            PortReceiver::new(
                receiver,
                port_id,
                /*coalesce=*/ true,
                self.state.clone(),
            ),
        )
    }

    /// Open a port that accepts M-typed messages, using the provided function
    /// to enqueue.
    // TODO: consider making lifetime bound to Self instead.
    pub(crate) fn open_enqueue_port<M: Message>(
        &self,
        enqueue: impl Fn(M) -> Result<(), anyhow::Error> + Send + Sync + 'static,
    ) -> PortHandle<M> {
        PortHandle {
            mailbox: self.clone(),
            port_index: self.state.allocate_port(),
            sender: UnboundedPortSender::Func(Arc::new(enqueue)),
            bound: Arc::new(OnceLock::new()),
        }
    }

    /// Open a new one-shot port that accepts M-typed messages. The
    /// returned port may be used to send a single message; ditto the
    /// receiver may receive a single message.
    pub fn open_once_port<M: Message>(&self) -> (OncePortHandle<M>, OncePortReceiver<M>) {
        let port_index = self.state.allocate_port();
        let port_id = PortId(self.state.actor_id.clone(), port_index);
        let (sender, receiver) = oneshot::channel::<M>();
        (
            OncePortHandle {
                mailbox: self.clone(),
                port_index,
                port_id: port_id.clone(),
                sender,
            },
            OncePortReceiver {
                receiver: Some(receiver),
                port_id,
                state: self.state.clone(),
            },
        )
    }

    fn error(&self, err: MailboxErrorKind) -> MailboxError {
        MailboxError::new(self.state.actor_id.clone(), err)
    }

    fn bind<M: RemoteMessage>(&self, handle: &PortHandle<M>) -> PortRef<M> {
        assert_eq!(
            handle.mailbox.actor_id(),
            self.actor_id(),
            "port does not belong to mailbox"
        );

        // TODO: don't even allocate a port until the port is bound. Possibly
        // have handles explicitly staged (unbound, bound).
        let port_id = self.actor_id().port_id(handle.port_index);
        match self.state.ports.entry(handle.port_index) {
            Entry::Vacant(entry) => {
                entry.insert(Box::new(UnboundedSender::new(
                    handle.sender.clone(),
                    port_id.clone(),
                )));
            }
            Entry::Occupied(_entry) => {}
        }

        PortRef::attest(port_id)
    }

    fn bind_to<M: RemoteMessage>(&self, handle: &PortHandle<M>, port_index: u64) {
        assert_eq!(
            handle.mailbox.actor_id(),
            self.actor_id(),
            "port does not belong to mailbox"
        );

        let port_id = self.actor_id().port_id(port_index);
        match self.state.ports.entry(port_index) {
            Entry::Vacant(entry) => {
                entry.insert(Box::new(UnboundedSender::new(
                    handle.sender.clone(),
                    port_id,
                )));
            }
            Entry::Occupied(_entry) => panic!("port {} already bound", port_id),
        }
    }

    fn bind_once<M: RemoteMessage>(&self, handle: OncePortHandle<M>) {
        let port_id = handle.port_id().clone();
        match self.state.ports.entry(handle.port_index) {
            Entry::Vacant(entry) => {
                entry.insert(Box::new(OnceSender::new(handle.sender, port_id.clone())));
            }
            Entry::Occupied(_entry) => {}
        }
    }
}

// TODO: figure out what to do with these interfaces -- possibly these caps
// do not have to be private.

/// Open a port given a capability.
pub fn open_port<M: Message>(caps: &impl cap::CanOpenPort) -> (PortHandle<M>, PortReceiver<M>) {
    caps.mailbox().open_port()
}

/// Open a one-shot port given a capability. This is a public method primarily to
/// enable macro-generated clients.
pub fn open_once_port<M: Message>(
    caps: &impl cap::CanOpenPort,
) -> (OncePortHandle<M>, OncePortReceiver<M>) {
    caps.mailbox().open_once_port()
}

impl MailboxSender for Mailbox {
    /// Deliver a serialized message to the provided port ID. This method fails
    /// if the message does not deserialize into the expected type.
    fn post(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        tracing::trace!(name = "post", "posting message to {}", envelope.dest);
        if envelope.dest().actor_id() != &self.state.actor_id {
            return self.state.forwarder.post(envelope, return_handle);
        }

        match self.state.ports.entry(envelope.dest().index()) {
            Entry::Vacant(_) => envelope.undeliverable(
                DeliveryError::Unroutable("port not bound in mailbox".to_string()),
                return_handle,
            ),
            Entry::Occupied(entry) => {
                let (metadata, data) = envelope.open();
                // We use the entry API here so that we can remove the
                // entry while holding an (entry) reference. The DashMap
                // documentation suggests that deadlocks are possible
                // "when holding any sort of reference into the map",
                // but surely this applies only to the same thread? This
                // would also imply we have to be careful holding any
                // sort of reference across .await points.
                match entry.get().send_serialized(data) {
                    Ok(false) => {
                        entry.remove();
                    }
                    Ok(true) => (),
                    Err(SerializedSenderError { data, error }) => MessageEnvelope::seal(
                        metadata, data,
                    )
                    .undeliverable(DeliveryError::Mailbox(format!("{}", error)), return_handle),
                }
            }
        }
    }
}

impl cap::sealed::CanSend for Mailbox {
    fn post(&self, dest: PortId, data: Serialized) {
        let envelope = MessageEnvelope::new(self.actor_id().clone(), dest, data);
        MailboxSender::post(self, envelope, monitored_return_handle());
    }
}

impl cap::sealed::CanOpenPort for Mailbox {
    fn mailbox(&self) -> &Mailbox {
        self
    }
}

/// A port to which M-typed messages can be delivered. Ports may be
/// serialized to be sent to other actors. However, when a port is
/// deserialized, it may no longer be used to send messages directly
/// to a mailbox since it is no longer associated with a local mailbox
/// ([`Mailbox::send`] will fail). However, the runtime may accept
/// remote Ports, and arrange for these messages to be delivered
/// indirectly through inter-node message passing.
#[derive(Debug)]
pub struct PortHandle<M: Message> {
    mailbox: Mailbox,
    // TEMPORARY until we remove hardcoded port IDs
    pub(crate) port_index: u64,
    sender: UnboundedPortSender<M>,
    // We would like this to be a Arc<OnceLock<PortRef<M>>>, but we cannot
    // write down the type PortRef<M> (M: Message), even though we cannot
    // legally construct such a value without M: RemoteMessage. We could consider
    // making PortRef<M> valid for M: Message, but constructible only for
    // M: RemoteMessage, but the guarantees offered by the impossibilty of even
    // writing down the type are appealing.
    bound: Arc<OnceLock<PortId>>,
}

impl<M: Message> PortHandle<M> {
    fn new(mailbox: Mailbox, port_index: u64, sender: UnboundedPortSender<M>) -> Self {
        Self {
            mailbox,
            port_index,
            sender,
            bound: Arc::new(OnceLock::new()),
        }
    }

    fn location(&self) -> PortLocation {
        match self.bound.get() {
            Some(port_id) => PortLocation::Bound(port_id.clone()),
            None => PortLocation::new_unbound::<M>(self.mailbox.actor_id().clone()),
        }
    }

    /// Send a message to this port.
    pub fn send(&self, message: M) -> Result<(), MailboxSenderError> {
        self.sender.send(message).map_err(|err| {
            MailboxSenderError::new_unbound::<M>(
                self.mailbox.actor_id().clone(),
                MailboxSenderErrorKind::Other(err),
            )
        })
    }
}

impl<M: RemoteMessage> PortHandle<M> {
    /// Bind this port, making it accessible to remote actors.
    pub fn bind(&self) -> PortRef<M> {
        PortRef::attest(
            self.bound
                .get_or_init(|| self.mailbox.bind(self).port_id().clone())
                .clone(),
        )
    }

    /// Bind to a specific port index. This is used by [`actor::Binder`] implementations to
    /// bind actor refs. This is not intended for general use.
    pub fn bind_to(&self, port_index: u64) {
        self.mailbox.bind_to(self, port_index);
    }
}

impl<M: Message> Clone for PortHandle<M> {
    fn clone(&self) -> Self {
        Self {
            mailbox: self.mailbox.clone(),
            port_index: self.port_index,
            sender: self.sender.clone(),
            bound: self.bound.clone(),
        }
    }
}

impl<M: Message> fmt::Display for PortHandle<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.location(), f)
    }
}

/// A one-shot port handle to which M-typed messages can be delivered.
#[derive(Debug)]
pub struct OncePortHandle<M: Message> {
    mailbox: Mailbox,
    port_index: u64,
    port_id: PortId,
    sender: oneshot::Sender<M>,
}

impl<M: Message> OncePortHandle<M> {
    /// This port's ID.
    // TODO: make value
    pub fn port_id(&self) -> &PortId {
        &self.port_id
    }

    /// Send a message to this port. The send operation will consume the
    /// port handle, as the port accepts at most one message.
    pub fn send(self, message: M) -> Result<(), MailboxSenderError> {
        let actor_id = self.mailbox.actor_id().clone();
        self.sender.send(message).map_err(|_| {
            // Here, the value is returned when the port is
            // closed.  We should consider having a similar
            // API for send_once, though arguably it makes less
            // sense in this context.
            MailboxSenderError::new_unbound::<M>(actor_id, MailboxSenderErrorKind::Closed)
        })?;
        Ok(())
    }
}

impl<M: RemoteMessage> OncePortHandle<M> {
    /// Turn this handle into a ref that may be passed to
    /// a remote actor. The remote actor can then use the
    /// ref to send a message to the port. Creating a ref also
    /// binds the port, so that it is remotely writable.
    pub fn bind(self) -> OncePortRef<M> {
        let port_id = self.port_id().clone();
        self.mailbox.clone().bind_once(self);
        OncePortRef::attest(port_id)
    }
}

impl<M: Message> fmt::Display for OncePortHandle<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.port_id(), f)
    }
}

/// A receiver of M-typed messages, used by actors to receive messages
/// on open ports.
#[derive(Debug)]
pub struct PortReceiver<M> {
    receiver: mpsc::UnboundedReceiver<M>,
    port_id: PortId,
    /// When multiple messages are put in channel, only receive the latest one
    /// if coalesce is true. Other messages will be discarded.
    coalesce: bool,
    /// State is used to remove the port from service when the receiver
    /// is dropped.
    state: Arc<State>,
}

impl<M> PortReceiver<M> {
    fn new(
        receiver: mpsc::UnboundedReceiver<M>,
        port_id: PortId,
        coalesce: bool,
        state: Arc<State>,
    ) -> Self {
        Self {
            receiver,
            port_id,
            coalesce,
            state,
        }
    }

    /// Tries to receive the next value for this receiver.
    /// This function returns `Ok(None)` if the receiver is empty
    /// and returns a MailboxError if the receiver is disconnected.
    pub fn try_recv(&mut self) -> Result<Option<M>, MailboxError> {
        let mut next = self.receiver.try_recv();
        // To coalesce, drain the mpsc queue and only keep the last one.
        if self.coalesce {
            if let Some(latest) = self.drain().pop() {
                next = Ok(latest);
            }
        }
        match next {
            Ok(msg) => Ok(Some(msg)),
            Err(mpsc::error::TryRecvError::Empty) => Ok(None),
            Err(mpsc::error::TryRecvError::Disconnected) => Err(MailboxError::new(
                self.actor_id().clone(),
                MailboxErrorKind::Closed,
            )),
        }
    }

    /// Receive the next message from the port corresponding with this
    /// receiver.
    pub async fn recv(&mut self) -> Result<M, MailboxError> {
        let mut next = self.receiver.recv().await;
        // To coalesce, get the last message from the queue if there are
        // more on the mspc queue.
        if self.coalesce {
            if let Some(latest) = self.drain().pop() {
                next = Some(latest);
            }
        }
        next.ok_or(MailboxError::new(
            self.actor_id().clone(),
            MailboxErrorKind::Closed,
        ))
    }

    /// Drains all available messages from the port.
    pub fn drain(&mut self) -> Vec<M> {
        let mut drained: Vec<M> = Vec::new();
        while let Ok(msg) = self.receiver.try_recv() {
            // To coalesce, discard the old message if there is any.
            if self.coalesce {
                drained.pop();
            }
            drained.push(msg);
        }
        drained
    }

    fn port(&self) -> u64 {
        self.port_id.1
    }

    fn actor_id(&self) -> &ActorId {
        &self.port_id.0
    }
}

impl<M> Drop for PortReceiver<M> {
    fn drop(&mut self) {
        // MARIUS: do we need to tombstone these? or should we
        // error out if we have removed the receiver before serializing the port ref?
        // ("no longer live")?
        self.state.ports.remove(&self.port());
    }
}

/// A receiver of M-typed messages from [`OncePort`]s.
pub struct OncePortReceiver<M> {
    receiver: Option<oneshot::Receiver<M>>,
    port_id: PortId,

    /// State is used to remove the port from service when the receiver
    /// is dropped.
    state: Arc<State>,
}

impl<M> OncePortReceiver<M> {
    /// Receive message from the one-shot port associated with this
    /// receiver.  Recv consumes the receiver: it is no longer valid
    /// after this call.
    pub async fn recv(mut self) -> Result<M, MailboxError> {
        std::mem::take(&mut self.receiver)
            .unwrap()
            .await
            .map_err(|err| {
                MailboxError::new(
                    self.actor_id().clone(),
                    MailboxErrorKind::Recv(self.port_id.clone(), err.into()),
                )
            })
    }

    fn port(&self) -> u64 {
        self.port_id.1
    }

    fn actor_id(&self) -> &ActorId {
        &self.port_id.0
    }
}

impl<M> Drop for OncePortReceiver<M> {
    fn drop(&mut self) {
        // MARIUS: do we need to tombstone these? or should we
        // error out if we have removed the receiver before serializing the port ref?
        // ("no longer live")?
        self.state.ports.remove(&self.port());
    }
}

/// Error that that occur during SerializedSender's send operation.
pub struct SerializedSenderError {
    /// The message was tried to send.
    pub data: Serialized,
    /// The mailbox sender error that occurred.
    pub error: MailboxSenderError,
}

/// SerializedSender encapsulates senders:
///   - It performs type erasure (and thus it is object-safe).
///   - It abstracts over [`Port`]s and [`OncePort`]s, by dynamically tracking the
///     validity of the underlying port.
trait SerializedSender: Send + Sync {
    /// Send a serialized message. SerializedSender will deserialize the
    /// message (failing if it fails to deserialize), and then send the
    /// resulting message on the underlying port.
    ///
    /// Send_serialized returns true whenever the port remains valid
    /// after the send operation.
    fn send_serialized(&self, serialized: Serialized) -> Result<bool, SerializedSenderError>;
}

/// A sender to an M-typed unbounded port.
enum UnboundedPortSender<M: Message> {
    /// Send directly to the mpsc queue.
    Mpsc(mpsc::UnboundedSender<M>),
    /// Use the provided function to enqueue the item.
    Func(Arc<dyn Fn(M) -> Result<(), anyhow::Error> + Send + Sync>),
}

impl<M: Message> UnboundedPortSender<M> {
    fn send(&self, message: M) -> Result<(), anyhow::Error> {
        match self {
            Self::Mpsc(sender) => sender.send(message).map_err(anyhow::Error::from),
            Self::Func(func) => func(message),
        }
    }
}

// We implement Clone manually as derive(Clone) places unnecessarily
// strict bounds on the type parameter M.
impl<M: Message> Clone for UnboundedPortSender<M> {
    fn clone(&self) -> Self {
        match self {
            Self::Mpsc(sender) => Self::Mpsc(sender.clone()),
            Self::Func(func) => Self::Func(func.clone()),
        }
    }
}

impl<M: Message> Debug for UnboundedPortSender<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Self::Mpsc(q) => f.debug_tuple("UnboundedPortSender::Mpsc").field(q).finish(),
            Self::Func(_) => f
                .debug_tuple("UnboundedPortSender::Func")
                .field(&"..")
                .finish(),
        }
    }
}

struct UnboundedSender<M: Message> {
    sender: UnboundedPortSender<M>,
    port_id: PortId,
}

impl<M: Message> UnboundedSender<M> {
    /// Create a new UnboundedSender encapsulating the provided
    /// sender.
    fn new(sender: UnboundedPortSender<M>, port_id: PortId) -> Self {
        Self { sender, port_id }
    }

    fn send(&self, message: M) -> Result<(), MailboxSenderError> {
        self.sender.send(message).map_err(|err| {
            MailboxSenderError::new_bound(self.port_id.clone(), MailboxSenderErrorKind::Other(err))
        })
    }
}

// Clone is implemented explicitly because the derive macro demands M:
// Clone directly. In this case, it isn't needed because Arc<T> can
// clone for any T.
impl<M: Message> Clone for UnboundedSender<M> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            port_id: self.port_id.clone(),
        }
    }
}

impl<M: RemoteMessage> SerializedSender for UnboundedSender<M> {
    fn send_serialized(&self, serialized: Serialized) -> Result<bool, SerializedSenderError> {
        match serialized.deserialized() {
            Ok(message) => {
                self.sender
                    .send(message)
                    .map_err(|err| SerializedSenderError {
                        data: serialized,
                        error: MailboxSenderError::new_bound(
                            self.port_id.clone(),
                            MailboxSenderErrorKind::Other(err),
                        ),
                    })?;

                Ok(true)
            }
            Err(err) => Err(SerializedSenderError {
                data: serialized,
                error: MailboxSenderError::new_bound(
                    self.port_id.clone(),
                    MailboxSenderErrorKind::Deserialize(M::typename(), err),
                ),
            }),
        }
    }
}

/// OnceSender encapsulates an underlying one-shot sender, dynamically
/// tracking its validity.
#[derive(Debug)]
struct OnceSender<M: Message> {
    sender: Arc<Mutex<Option<oneshot::Sender<M>>>>,
    port_id: PortId,
}

impl<M: Message> OnceSender<M> {
    /// Create a new OnceSender encapsulating the provided one-shot
    /// sender.
    fn new(sender: oneshot::Sender<M>, port_id: PortId) -> Self {
        Self {
            sender: Arc::new(Mutex::new(Some(sender))),
            port_id,
        }
    }

    fn send_once(&self, message: M) -> Result<bool, MailboxSenderError> {
        // TODO: we should replace the sender on error
        match self.sender.lock().unwrap().take() {
            None => Err(MailboxSenderError::new_bound(
                self.port_id.clone(),
                MailboxSenderErrorKind::Closed,
            )),
            Some(sender) => {
                sender.send(message).map_err(|_| {
                    // Here, the value is returned when the port is
                    // closed.  We should consider having a similar
                    // API for send_once, though arguably it makes less
                    // sense in this context.
                    MailboxSenderError::new_bound(
                        self.port_id.clone(),
                        MailboxSenderErrorKind::Closed,
                    )
                })?;
                Ok(false)
            }
        }
    }
}

// Clone is implemented explicitly because the derive macro demands M:
// Clone directly. In this case, it isn't needed because Arc<T> can
// clone for any T.
impl<M: Message> Clone for OnceSender<M> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            port_id: self.port_id.clone(),
        }
    }
}

impl<M: RemoteMessage> SerializedSender for OnceSender<M> {
    fn send_serialized(&self, serialized: Serialized) -> Result<bool, SerializedSenderError> {
        match serialized.deserialized() {
            Ok(message) => self.send_once(message).map_err(|e| SerializedSenderError {
                data: serialized,
                error: e,
            }),
            Err(err) => Err(SerializedSenderError {
                data: serialized,
                error: MailboxSenderError::new_bound(
                    self.port_id.clone(),
                    MailboxSenderErrorKind::Deserialize(M::typename(), err),
                ),
            }),
        }
    }
}

/// State is the internal state of the mailbox.
struct State {
    /// The ID of the mailbox owner.
    actor_id: ActorId,

    // insert if it's serializable; otherwise don't.
    /// The set of active ports in the mailbox. All currently
    /// allocated ports are
    ports: DashMap<u64, Box<dyn SerializedSender>>,

    /// The next port ID to allocate.
    next_port: AtomicU64,

    /// The forwarder for this mailbox.
    forwarder: BoxedMailboxSender,
}

impl State {
    /// Create a new state with the provided owning ActorId.
    fn new(actor_id: ActorId, forwarder: BoxedMailboxSender) -> Self {
        Self {
            actor_id,
            ports: DashMap::new(),
            // The first 1024 ports are allocated to actor handlers.
            // Other port IDs are ephemeral.
            next_port: AtomicU64::new(USER_PORT_OFFSET),
            forwarder,
        }
    }

    /// Allocate a fresh port.
    fn allocate_port(&self) -> u64 {
        self.next_port.fetch_add(1, Ordering::SeqCst)
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_struct("State")
            .field("actor_id", &self.actor_id)
            .field(
                "open_ports",
                &self.ports.iter().map(|e| *e.key()).collect::<Vec<_>>(),
            )
            .field("next_port", &self.next_port)
            .finish()
    }
}

// TODO: mux based on some parameterized type. (mux key).
/// An in-memory mailbox muxer. This is used to route messages to
/// different underlying senders.
#[derive(Debug, Clone)]
pub struct MailboxMuxer {
    mailboxes: Arc<DashMap<ActorId, Box<dyn MailboxSender + Send + Sync>>>,
}

impl MailboxMuxer {
    /// Create a new, empty, muxer.
    pub fn new() -> Self {
        Self {
            mailboxes: Arc::new(DashMap::new()),
        }
    }

    /// Route messages destined for the provided actor id to the provided
    /// sender. Returns false if there is already a sender associated
    /// with the actor. In this case, the sender is not replaced, and
    /// the caller must [`MailboxMuxer::unbind`] it first.
    pub fn bind(&self, actor_id: ActorId, sender: impl MailboxSender + 'static) -> bool {
        match self.mailboxes.entry(actor_id) {
            Entry::Occupied(_) => false,
            Entry::Vacant(entry) => {
                entry.insert(Box::new(sender));
                true
            }
        }
    }

    /// Convenience function to bind a mailbox.
    pub fn bind_mailbox(&self, mailbox: Mailbox) -> bool {
        self.bind(mailbox.actor_id().clone(), mailbox)
    }

    /// Unbind the sender associated with the provided actor ID. After
    /// unbinding, the muxer will no longer be able to send messages to
    /// that actor.
    pub(crate) fn unbind(&self, actor_id: &ActorId) {
        self.mailboxes.remove(actor_id);
    }

    /// Returns a list of all actors bound to this muxer. Useful in debugging.
    pub fn bound_actors(&self) -> Vec<ActorId> {
        self.mailboxes.iter().map(|e| e.key().clone()).collect()
    }
}

impl MailboxSender for MailboxMuxer {
    fn post(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        let dest_actor_id = envelope.dest().actor_id();
        match self.mailboxes.get(envelope.dest().actor_id()) {
            None => {
                let err = format!("no mailbox for actor {} registered in muxer", dest_actor_id);
                envelope.undeliverable(DeliveryError::Unroutable(err), return_handle)
            }
            Some(sender) => sender.post(envelope, return_handle),
        }
    }
}

/// MailboxRouter routes messages to the sender that is bound to its
/// nearest prefix.
#[derive(Debug, Clone)]
pub struct MailboxRouter {
    entries: Arc<RwLock<BTreeMap<Reference, Arc<dyn MailboxSender + Send + Sync>>>>,
}

impl MailboxRouter {
    /// Create a new, empty router.
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    /// Downgrade this router to a [`WeakMailboxRouter`].
    pub fn downgrade(&self) -> WeakMailboxRouter {
        WeakMailboxRouter(Arc::downgrade(&self.entries))
    }

    /// Bind the provided sender to the given reference. The destination
    /// is treated as a prefix to which messages can be routed, and
    /// messages are routed to their longest matching prefix.
    pub fn bind(&self, dest: Reference, sender: impl MailboxSender + 'static) {
        let mut w = self.entries.write().unwrap();
        w.insert(dest, Arc::new(sender));
    }
}

impl MailboxSender for MailboxRouter {
    fn post(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        let sender = {
            let actor_id = envelope.dest().actor_id();
            match self
                .entries
                .read()
                .unwrap()
                .lower_bound(Excluded(&actor_id.clone().into()))
                .prev()
            {
                None => None,
                Some((key, sender)) if key.is_prefix_of(&actor_id.clone().into()) => {
                    Some(sender.clone())
                }
                Some(_) => None,
            }
        };

        match sender {
            None => envelope.undeliverable(
                DeliveryError::Unroutable(
                    "no destination found for actor in routing table".to_string(),
                ),
                return_handle,
            ),
            Some(sender) => sender.post(envelope, return_handle),
        }
    }
}

/// A version of [`MailboxRouter`] that holds a weak reference to the underlying
/// state. This allows router references to be circular: an entity holding a reference
/// to the router may also contain the router itself.
///
/// TODO: this currently holds a weak reference to the entire router. This helps
/// prevent cycle leaks, but can cause excess memory usage as the cycle is at
/// the granularity of each entry. Possibly the router should allow weak references
/// on a per-entry basis.
#[derive(Debug, Clone)]
pub struct WeakMailboxRouter(
    Weak<RwLock<BTreeMap<Reference, Arc<dyn MailboxSender + Send + Sync>>>>,
);

impl WeakMailboxRouter {
    /// Upgrade the weak router to a strong reference router.
    pub fn upgrade(&self) -> Option<MailboxRouter> {
        self.0.upgrade().map(|entries| MailboxRouter { entries })
    }
}

impl MailboxSender for WeakMailboxRouter {
    fn post(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        match self.upgrade() {
            Some(router) => router.post(envelope, return_handle),
            None => envelope.undeliverable(
                DeliveryError::BrokenLink("failed to upgrade WeakMailboxRouter".to_string()),
                return_handle,
            ),
        }
    }
}

/// A serializable [`MailboxRouter`]. It keeps a serializable address book so that
/// the mailbox sender can be recovered.
#[derive(Debug, Clone)]
pub struct DialMailboxRouter {
    address_book: Arc<RwLock<BTreeMap<Reference, ChannelAddr>>>,
    sender_cache: Arc<DashMap<ChannelAddr, Arc<MailboxClient>>>,

    // The default sender, to which messages for unknown recipients
    // are sent. (This is like a default route in a routing table.)
    default: BoxedMailboxSender,
}

impl DialMailboxRouter {
    /// Create a new [`DialMailboxRouter`] with an empty routing table.
    pub fn new() -> Self {
        Self::new_with_default(BoxedMailboxSender::new(UnroutableMailboxSender))
    }

    /// Create a new [`DialMailboxRouter`] with an empty routing table,
    /// and a default sender. Any message with an unknown destination is
    /// dispatched on this default sender.
    pub fn new_with_default(default: BoxedMailboxSender) -> Self {
        Self {
            address_book: Arc::new(RwLock::new(BTreeMap::new())),
            sender_cache: Arc::new(DashMap::new()),
            default,
        }
    }

    /// Binds a [`Reference`] to a [`ChannelAddr`], replacing any
    /// existing binding.
    ///
    /// If the address changes, the old sender is evicted from the
    /// cache to ensure fresh routing on next use.
    pub fn bind(&self, dest: Reference, addr: ChannelAddr) {
        if let Ok(mut w) = self.address_book.write() {
            if let Some(old_addr) = w.insert(dest.clone(), addr.clone()) {
                if old_addr != addr {
                    tracing::info!("Rebinding {:?} from {:?} to {:?}", dest, old_addr, addr);
                    self.sender_cache.remove(&old_addr);
                }
            }
        } else {
            tracing::error!("Address book poisoned during bind of {:?}", dest);
        }
    }

    /// Removes all address mappings with the given prefix from the
    /// router.
    ///
    /// Also evicts any corresponding cached senders to prevent reuse
    /// of stale connections.
    pub fn unbind(&self, dest: &Reference) {
        if let Ok(mut w) = self.address_book.write() {
            let to_remove: Vec<(Reference, ChannelAddr)> = w
                .range(dest..)
                .take_while(|(key, _)| dest.is_prefix_of(key))
                .map(|(key, addr)| (key.clone(), addr.clone()))
                .collect();

            for (key, addr) in to_remove {
                tracing::info!("Unbinding {:?} from {:?}", key, addr);
                w.remove(&key);
                self.sender_cache.remove(&addr);
            }
        } else {
            tracing::error!("Address book poisoned during unbind of {:?}", dest);
        }
    }

    /// Lookup an actor's channel in the router's address bok.
    pub fn lookup_addr(&self, actor_id: &ActorId) -> Option<ChannelAddr> {
        let address_book = self.address_book.read().unwrap();
        address_book
            .lower_bound(Excluded(&actor_id.clone().into()))
            .prev()
            .and_then(|(key, addr)| {
                if key.is_prefix_of(&actor_id.clone().into()) {
                    Some(addr.clone())
                } else {
                    None
                }
            })
    }

    fn dial(
        &self,
        addr: &ChannelAddr,
        actor_id: &ActorId,
    ) -> Result<Arc<MailboxClient>, MailboxSenderError> {
        // Get the sender. Create it if needed. Do not send the
        // messages inside this block so we do not hold onto the
        // reference of the dashmap entries.
        match self.sender_cache.entry(addr.clone()) {
            Entry::Occupied(entry) => Ok(entry.get().clone()),
            Entry::Vacant(entry) => {
                let tx = channel::dial(addr.clone()).map_err(|err| {
                    MailboxSenderError::new_unbound_type(
                        actor_id.clone(),
                        MailboxSenderErrorKind::Channel(err),
                        "unknown",
                    )
                })?;
                let sender = MailboxClient::new(tx);
                Ok(entry.insert(Arc::new(sender)).value().clone())
            }
        }
    }
}

impl MailboxSender for DialMailboxRouter {
    fn post(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        let Some(addr) = self.lookup_addr(envelope.dest().actor_id()) else {
            self.default.post(envelope, return_handle);
            return;
        };

        match self.dial(&addr, envelope.dest().actor_id()) {
            Err(err) => envelope.undeliverable(
                DeliveryError::Unroutable(format!("cannot dial destination: {err}")),
                return_handle,
            ),
            Ok(sender) => sender.post(envelope, return_handle),
        }
    }
}

/// A MailboxSender that reports any envelope as undeliverable due to
/// routing failure.
#[derive(Debug)]
pub struct UnroutableMailboxSender;

impl MailboxSender for UnroutableMailboxSender {
    fn post(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        envelope.undeliverable(
            DeliveryError::Unroutable("destination not found in routing table".to_string()),
            return_handle,
        );
    }
}

#[cfg(test)]
mod tests {

    use std::assert_matches::assert_matches;
    use std::mem::drop;
    use std::sync::atomic::AtomicUsize;

    use super::*;
    use crate::Actor;
    use crate::PortId;
    use crate::accum;
    use crate::channel::ChannelTransport;
    use crate::channel::sim;
    use crate::channel::sim::SimAddr;
    use crate::clock::Clock;
    use crate::clock::RealClock;
    use crate::data::Serialized;
    use crate::id;
    use crate::proc::Proc;
    use crate::reference::ProcId;
    use crate::reference::WorldId;

    #[test]
    fn test_error() {
        let err = MailboxError::new(
            ActorId(ProcId(WorldId("myworld".into()), 2), "myactor".into(), 5),
            MailboxErrorKind::Closed,
        );
        assert_eq!(format!("{}", err), "myworld[2].myactor[5]: mailbox closed");
    }

    #[tokio::test]
    async fn test_mailbox_basic() {
        let mbox = Mailbox::new_detached(id!(test[0].test));
        let (port, mut receiver) = mbox.open_port::<u64>();
        let port = port.bind();

        mbox.serialize_and_send(&port, 123, monitored_return_handle())
            .unwrap();
        mbox.serialize_and_send(&port, 321, monitored_return_handle())
            .unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 123u64);
        assert_eq!(receiver.recv().await.unwrap(), 321u64);

        let serialized = Serialized::serialize(&999u64).unwrap();
        mbox.post(
            MessageEnvelope::new_unknown(port.port_id().clone(), serialized),
            monitored_return_handle(),
        );
        assert_eq!(receiver.recv().await.unwrap(), 999u64);
    }

    #[tokio::test]
    async fn test_mailbox_accum() {
        let mbox = Mailbox::new_detached(id!(test[0].test));
        let (port, mut receiver) = mbox.open_accum_port(accum::max::<u64>());

        port.send(1).unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 1);
        port.send(2).unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 2);
        port.send(3).unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 3);
        // Send a smaller value. Should still receive the previous max.
        port.send(1).unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 3);
        port.send(2).unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 3);
        port.send(3).unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 3);
        port.send(4).unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 4);
        // Send multiple updates. Should only receive the final change.
        port.send(5).unwrap();
        port.send(6).unwrap();
        port.send(7).unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 7);
        port.send(1).unwrap();
        port.send(3).unwrap();
        port.send(2).unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 7);
    }

    #[tokio::test]
    #[ignore] // error behavior changed, but we will bring it back
    async fn test_mailbox_once() {
        let mbox = Mailbox::new_detached(id!(test[0].test));

        let (port, receiver) = mbox.open_once_port::<u64>();

        // let port_id = port.port_id().clone();

        port.send(123u64).unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 123u64);

        // // The borrow checker won't let us send again on the port
        // // (good!), but we stashed the port-id and so we can try on the
        // // serialized interface.
        // let Err(err) = mbox
        //     .send_serialized(&port_id, &Serialized(Vec::new()))
        //     .await
        // else {
        //     unreachable!()
        // };
        // assert_matches!(err.kind(), MailboxSenderErrorKind::Closed);
    }

    #[tokio::test]
    #[ignore] // changed error behavior
    async fn test_mailbox_receiver_drop() {
        let mbox = Mailbox::new_detached(id!(test[0].test));
        let (port, mut receiver) = mbox.open_port::<u64>();
        // Make sure we go through "remote" path.
        let port = port.bind();
        mbox.serialize_and_send(&port, 123u64, monitored_return_handle())
            .unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 123u64);
        drop(receiver);
        let Err(err) = mbox.serialize_and_send(&port, 123u64, monitored_return_handle()) else {
            panic!();
        };

        assert_matches!(err.kind(), MailboxSenderErrorKind::Closed);
        assert_matches!(err.location(), PortLocation::Bound(bound) if bound == port.port_id());
    }

    #[tokio::test]
    async fn test_drain() {
        let mbox = Mailbox::new_detached(id!(test[0].test));

        let (port, mut receiver) = mbox.open_port();
        let port = port.bind();

        for i in 0..10 {
            mbox.serialize_and_send(&port, i, monitored_return_handle())
                .unwrap();
        }

        for i in 0..10 {
            assert_eq!(receiver.recv().await.unwrap(), i);
        }

        assert!(receiver.drain().is_empty());
    }

    #[tokio::test]
    async fn test_mailbox_muxer() {
        let muxer = MailboxMuxer::new();

        let mbox0 = Mailbox::new_detached(id!(test[0].actor1));
        let mbox1 = Mailbox::new_detached(id!(test[0].actor2));

        muxer.bind(mbox0.actor_id().clone(), mbox0.clone());
        muxer.bind(mbox1.actor_id().clone(), mbox1.clone());

        let (port, receiver) = mbox0.open_once_port::<u64>();

        port.send(123u64).unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 123u64);

        /*
        let (tx, rx) = channel::local::new::<u64>();
        let (port, _) = mbox0.open_port::<u64>();
        let handle = muxer.clone().serve_port(port, rx).unwrap();
        muxer.unbind(mbox0.actor_id());
        tx.send(123u64).await.unwrap();
        let Ok(Err(err)) = handle.await else { panic!() };
        assert_eq!(err.actor_id(), &actor_id(0));
        */
    }

    #[tokio::test]
    async fn test_local_client_server() {
        let mbox = Mailbox::new_detached(id!(test[0].actor0));
        let (tx, rx) = channel::local::new();
        let serve_handle = mbox.clone().serve(rx, monitored_return_handle());
        let client = MailboxClient::new(tx);

        let (port, receiver) = mbox.open_once_port::<u64>();
        let port = port.bind();

        client
            .serialize_and_send_once(port, 123u64, monitored_return_handle())
            .unwrap();
        assert_eq!(receiver.recv().await.unwrap(), 123u64);
        serve_handle.stop();
        serve_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_sim_client_server() {
        let src_addr = "local!0".parse::<ChannelAddr>().unwrap();
        let dst_addr = "local!1".parse::<ChannelAddr>().unwrap();

        let proxy = ChannelAddr::any(channel::ChannelTransport::Unix);
        let sim_addr = SimAddr::new(
            src_addr.clone(),
            proxy.clone(),
            dst_addr.clone(),
            proxy.clone(),
        )
        .unwrap();

        let (_, rx) = sim::serve::<MessageEnvelope>(sim_addr.reversed().clone()).unwrap();
        let tx = sim::dial::<MessageEnvelope>(sim_addr).unwrap();
        let mbox = Mailbox::new_detached(id!(test[0].actor0));
        let serve_handle = mbox.clone().serve(rx, monitored_return_handle());
        let client = MailboxClient::new(tx);
        let (port, receiver) = mbox.open_once_port::<u64>();
        let port = port.bind();
        let msg: u64 = 123;
        client
            .serialize_and_send_once(port, msg, monitored_return_handle())
            .unwrap();
        assert_eq!(receiver.recv().await.unwrap(), msg);
        serve_handle.stop();
        serve_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_mailbox_router() {
        let mbox0 = Mailbox::new_detached(id!(world0[0].actor0));
        let mbox1 = Mailbox::new_detached(id!(world1[0].actor0));
        let mbox2 = Mailbox::new_detached(id!(world1[1].actor0));
        let mbox3 = Mailbox::new_detached(id!(world1[1].actor1));

        let comms: Vec<(OncePortRef<u64>, OncePortReceiver<u64>)> =
            [&mbox0, &mbox1, &mbox2, &mbox3]
                .into_iter()
                .map(|mbox| {
                    let (port, receiver) = mbox.open_once_port::<u64>();
                    (port.bind(), receiver)
                })
                .collect();

        let router = MailboxRouter::new();

        router.bind(id!(world0).into(), mbox0);
        router.bind(id!(world1[0]).into(), mbox1);
        router.bind(id!(world1[1]).into(), mbox2);
        router.bind(id!(world1[1].actor1).into(), mbox3);

        for (i, (port, receiver)) in comms.into_iter().enumerate() {
            router
                .serialize_and_send_once(port, i as u64, monitored_return_handle())
                .unwrap();
            assert_eq!(receiver.recv().await.unwrap(), i as u64);
        }
    }

    #[tokio::test]
    async fn test_dial_mailbox_router() {
        let router = DialMailboxRouter::new();

        router.bind(id!(world0[0]).into(), "unix!@1".parse().unwrap());
        router.bind(id!(world1[0]).into(), "unix!@2".parse().unwrap());
        router.bind(id!(world1[1]).into(), "unix!@3".parse().unwrap());
        router.bind(id!(world1[1].actor1).into(), "unix!@4".parse().unwrap());

        // We should be able to lookup the ids
        router.lookup_addr(&id!(world0[0].actor[0])).unwrap();
        router.lookup_addr(&id!(world1[0].actor[0])).unwrap();

        // Unbind so we cannot find the ids anymore
        router.unbind(&id!(world1).into());
        assert!(router.lookup_addr(&id!(world1[0].actor1[0])).is_none());
        assert!(router.lookup_addr(&id!(world1[1].actor1[0])).is_none());
        assert!(router.lookup_addr(&id!(world1[2].actor1[0])).is_none());
        router.lookup_addr(&id!(world0[0].actor[0])).unwrap();
        router.unbind(&id!(world0).into());
        assert!(router.lookup_addr(&id!(world0[0].actor[0])).is_none());
    }

    #[tokio::test]
    #[ignore] // TODO: there's a leak here, fix it
    async fn test_dial_mailbox_router_default() {
        let mbox0 = Mailbox::new_detached(id!(world0[0].actor0));
        let mbox1 = Mailbox::new_detached(id!(world1[0].actor0));
        let mbox2 = Mailbox::new_detached(id!(world1[1].actor0));
        let mbox3 = Mailbox::new_detached(id!(world1[1].actor1));

        // We don't need to dial here, since we gain direct access to the
        // underlying routers.
        let root = MailboxRouter::new();
        let world0_router = DialMailboxRouter::new_with_default(root.boxed());
        let world1_router = DialMailboxRouter::new_with_default(root.boxed());

        root.bind(id!(world0).into(), world0_router.clone());
        root.bind(id!(world1).into(), world1_router.clone());

        let mailboxes = [&mbox0, &mbox1, &mbox2, &mbox3];

        let mut handles = Vec::new(); // hold on to handles, or channels get closed
        for mbox in mailboxes.iter() {
            let (addr, rx) = channel::serve(ChannelAddr::any(ChannelTransport::Local))
                .await
                .unwrap();
            let handle = (*mbox).clone().serve(rx, monitored_return_handle());
            handles.push(handle);

            eprintln!("{}: {}", mbox.actor_id(), addr);
            if mbox.actor_id().world_name() == "world0" {
                world0_router.bind(mbox.actor_id().clone().into(), addr);
            } else {
                world1_router.bind(mbox.actor_id().clone().into(), addr);
            }
        }

        // Make sure nodes are fully connected.
        for router in [root.boxed(), world0_router.boxed(), world1_router.boxed()] {
            for mbox in mailboxes.iter() {
                let (port, receiver) = mbox.open_once_port::<u64>();
                let port = port.bind();
                router
                    .serialize_and_send_once(port, 123u64, monitored_return_handle())
                    .unwrap();
                assert_eq!(receiver.recv().await.unwrap(), 123u64);
            }
        }
    }

    #[tokio::test]
    async fn test_enqueue_port() {
        let mbox = Mailbox::new_detached(id!(test[0].test));

        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();
        let port = mbox.open_enqueue_port(move |n| {
            count_clone.fetch_add(n, Ordering::SeqCst);
            Ok(())
        });

        port.send(10).unwrap();
        port.send(5).unwrap();
        port.send(1).unwrap();
        port.send(0).unwrap();

        assert_eq!(count.load(Ordering::SeqCst), 16);
    }

    #[derive(Clone, Debug, Serialize, Deserialize, Named)]
    struct TestMessage;

    #[derive(Clone, Debug, Serialize, Deserialize, Named)]
    #[named(name = "some::custom::path")]
    struct TestMessage2;

    #[test]
    fn test_remote_message_macros() {
        assert_eq!(
            TestMessage::typename(),
            "hyperactor::mailbox::tests::TestMessage"
        );
        assert_eq!(TestMessage2::typename(), "some::custom::path");
    }

    #[test]
    fn test_message_envelope_display() {
        #[derive(Named, Serialize, Deserialize)]
        struct MyTest {
            a: u64,
            b: String,
        }

        let envelope = MessageEnvelope::serialize(
            id!(source[0].actor),
            id!(dest[1].actor[0][123]),
            &MyTest {
                a: 123,
                b: "hello".into(),
            },
        )
        .unwrap();

        assert_eq!(
            format!("{}", envelope),
            r#"source[0].actor[0] > dest[1].actor[0][123]: MyTest{"a":123,"b":"hello"}"#
        );
    }

    #[derive(Debug)]
    struct Foo;

    #[async_trait]
    impl Actor for Foo {
        type Params = ();
        async fn new(_params: ()) -> Result<Self, anyhow::Error> {
            Ok(Self)
        }
    }

    // Test that a message delivery failure causes the sending actor
    // to stop running.
    #[tokio::test]
    async fn test_actor_delivery_failure() {
        // This test involves making an actor fail and so we must set
        // a supervision coordinator.
        use crate::actor::ActorStatus;
        use crate::test_utils::proc_supervison::ProcSupervisionCoordinator;

        let proc_forwarder = BoxedMailboxSender::new(DialMailboxRouter::new_with_default(
            BOXED_PANICKING_MAILBOX_SENDER.clone(),
        ));
        let proc_id = id!(quux[0]);
        let mut proc = Proc::new(proc_id.clone(), proc_forwarder);
        ProcSupervisionCoordinator::set(&proc).await.unwrap();

        let foo = proc.spawn::<Foo>("foo", ()).await.unwrap();
        let return_handle = foo.port::<Undeliverable<MessageEnvelope>>();
        let message = MessageEnvelope::new(
            foo.actor_id().clone(),
            PortId(id!(corge[0].bar), 9999u64),
            Serialized::serialize(&1u64).unwrap(),
        );
        return_handle.send(Undeliverable(message)).unwrap();

        RealClock
            .sleep(tokio::time::Duration::from_millis(100))
            .await;

        let foo_status = foo.status();
        assert!(matches!(*foo_status.borrow(), ActorStatus::Failed(_)));
        let ActorStatus::Failed(ref msg) = *foo_status.borrow() else {
            unreachable!()
        };
        assert!(msg.as_str().contains(
            "serving quux[0].foo[0]: processing error: a message from \
                quux[0].foo[0] to corge[0].bar[0][9999] was undeliverable and returned"
        ));

        proc.destroy_and_wait(tokio::time::Duration::from_secs(1), None)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_detached_return_handle() {
        let (return_handle, mut return_receiver) = new_undeliverable_port();
        // Simulate an undelivered message return.
        let envelope = MessageEnvelope::new(
            id!(foo[0].bar),
            PortId(id!(baz[0].corge), 9999u64),
            Serialized::serialize(&1u64).unwrap(),
        );
        return_handle.send(Undeliverable(envelope.clone())).unwrap();
        // Check we receive the undelivered message.
        assert!(
            tokio::time::timeout(tokio::time::Duration::from_secs(1), return_receiver.recv())
                .await
                .is_ok()
        );
        // Setup a monitor for the receiver and show that if there are
        // no outstanding return handles it terminates.
        let monitor_handle = tokio::spawn(async move {
            while let Ok(Undeliverable(mut envelope)) = return_receiver.recv().await {
                envelope.try_set_error(DeliveryError::BrokenLink(
                    "returned in unit test".to_string(),
                ));
                UndeliverableMailboxSender
                    .post(envelope, /*unused */ monitored_return_handle().clone());
            }
        });
        drop(return_handle);
        assert!(
            tokio::time::timeout(tokio::time::Duration::from_secs(1), monitor_handle)
                .await
                .is_ok()
        );
    }

    async fn verify_receiver(coalesce: bool, drop_sender: bool) {
        fn create_receiver<M>(coalesce: bool) -> (mpsc::UnboundedSender<M>, PortReceiver<M>) {
            // Create dummy state and port_id to create PortReceiver. They are
            // not used in the test.
            let dummy_state =
                State::new(id!(world[0].actor), BOXED_PANICKING_MAILBOX_SENDER.clone());
            let dummy_port_id = PortId(id!(world[0].actor), 0);
            let (sender, receiver) = mpsc::unbounded_channel::<M>();
            let receiver = PortReceiver {
                receiver,
                port_id: dummy_port_id,
                coalesce,
                state: Arc::new(dummy_state),
            };
            (sender, receiver)
        }

        // verify fn drain
        {
            let (sender, mut receiver) = create_receiver::<u64>(coalesce);
            assert!(receiver.drain().is_empty());

            sender.send(0).unwrap();
            sender.send(1).unwrap();
            sender.send(2).unwrap();
            sender.send(3).unwrap();
            sender.send(4).unwrap();
            sender.send(5).unwrap();
            sender.send(6).unwrap();
            sender.send(7).unwrap();

            if drop_sender {
                drop(sender);
            }

            if !coalesce {
                assert_eq!(receiver.drain(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
            } else {
                assert_eq!(receiver.drain(), vec![7]);
            }

            assert!(receiver.drain().is_empty());
            assert!(receiver.drain().is_empty());
        }

        // verify fn try_recv
        {
            let (sender, mut receiver) = create_receiver::<u64>(coalesce);
            assert!(receiver.try_recv().unwrap().is_none());

            sender.send(0).unwrap();
            sender.send(1).unwrap();
            sender.send(2).unwrap();
            sender.send(3).unwrap();

            if drop_sender {
                drop(sender);
            }

            if !coalesce {
                assert_eq!(receiver.try_recv().unwrap().unwrap(), 0);
                assert_eq!(receiver.try_recv().unwrap().unwrap(), 1);
                assert_eq!(receiver.try_recv().unwrap().unwrap(), 2);
            }
            assert_eq!(receiver.try_recv().unwrap().unwrap(), 3);
            if drop_sender {
                assert_matches!(
                    receiver.try_recv().unwrap_err().kind(),
                    MailboxErrorKind::Closed
                );
                // Still Closed error
                assert_matches!(
                    receiver.try_recv().unwrap_err().kind(),
                    MailboxErrorKind::Closed
                );
            } else {
                assert!(receiver.try_recv().unwrap().is_none());
                // Still empty
                assert!(receiver.try_recv().unwrap().is_none());
            }
        }
        // verify fn recv
        {
            let (sender, mut receiver) = create_receiver::<u64>(coalesce);
            assert!(
                tokio::time::timeout(tokio::time::Duration::from_secs(1), receiver.recv())
                    .await
                    .is_err()
            );

            sender.send(4).unwrap();
            sender.send(5).unwrap();
            sender.send(6).unwrap();
            sender.send(7).unwrap();

            if drop_sender {
                drop(sender);
            }

            if !coalesce {
                assert_eq!(receiver.recv().await.unwrap(), 4);
                assert_eq!(receiver.recv().await.unwrap(), 5);
                assert_eq!(receiver.recv().await.unwrap(), 6);
            }
            assert_eq!(receiver.recv().await.unwrap(), 7);
            if drop_sender {
                assert_matches!(
                    receiver.recv().await.unwrap_err().kind(),
                    MailboxErrorKind::Closed
                );
                // Still None
                assert_matches!(
                    receiver.recv().await.unwrap_err().kind(),
                    MailboxErrorKind::Closed
                );
            } else {
                assert!(
                    tokio::time::timeout(tokio::time::Duration::from_secs(1), receiver.recv())
                        .await
                        .is_err()
                );
            }
        }
    }

    #[tokio::test]
    async fn test_receiver_basic_default() {
        verify_receiver(/*coalesce=*/ false, /*drop_sender=*/ false).await
    }

    #[tokio::test]
    async fn test_receiver_basic_latest() {
        verify_receiver(/*coalesce=*/ true, /*drop_sender=*/ false).await
    }

    #[tokio::test]
    async fn test_receiver_after_sender_drop_default() {
        verify_receiver(/*coalesce=*/ false, /*drop_sender=*/ true).await
    }

    #[tokio::test]
    async fn test_receiver_after_sender_drop_latest() {
        verify_receiver(/*coalesce=*/ true, /*drop_sender=*/ true).await
    }
}
