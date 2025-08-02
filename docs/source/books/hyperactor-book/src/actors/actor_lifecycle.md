# Actor Lifecycle Types

This page documents auxiliary types used in actor startup, shutdown, and supervision logic.

## `ActorStatus`

`ActorStatus` describes the current runtime state of an actor. It is used to monitor progress, coordinate shutdown, and detect failure conditions.
```rust
pub enum ActorStatus {
    Unknown,
    Created,
    Initializing,
    Client,
    Idle,
    Processing(SystemTime, Option<(String, Option<String>)>),
    Saving(SystemTime),
    Loading(SystemTime),
    Stopping,
    Stopped,
    Failed(String),
}
```

### States
- `Unknown`: The status is unknown (e.g. not yet initialized).
- `Created`: The actor has been constructed but not yet started.
- `Initializing`: The actor is running its init lifecycle hook and is not yet receiving messages.
- `Client`: The actor is operating in “client” mode; its ports are being managed manually.
- `Idle`: The actor is ready to process messages but is currently idle.
- `Processing`: The actor is handling a message. Contains a timestamp and optionally the handler/arm label.
- `Saving`: The actor is saving its state as part of a checkpoint. Includes the time the operation began.
- `Loading`: The actor is loading a previously saved state.
- `Stopping`: The actor is in shutdown mode and draining its mailbox.
- `Stopped`: The actor has exited and will no longer process messages.
- `Failed`: The actor terminated abnormally. Contains an error description.

### Methods
- `is_terminal(&self) -> bool`: Returns true if the actor has either stopped or failed.
- `is_failed(&self) -> bool`: Returns true if the actor is in the Failed state.
- `passthrough(&self) -> ActorStatus`: Returns a clone of the status. Used internally during joins.
- `span_string(&self) -> &'static str`: Returns the active handler/arm name if available. Used for tracing.

## `Signal`

`Signal` is used to control actor lifecycle transitions externally. These messages are sent internally by the runtime (or explicitly by users) to initiate operations like shutdown.
```rust
pub enum Signal {
    Stop,
    DrainAndStop,
    Save,
    Load,
}
```
Variants
- `Stop`: Immediately halts the actor, even if messages remain in its mailbox.
- `DrainAndStop`: Gracefully stops the actor by first draining all queued messages.
- `Save`: Triggers a state snapshot using the actor’s Checkpointable::save method.
- `Load`: Requests state restoration via Checkpointable::load.

These signals are routed like any other message, typically sent using `ActorHandle::send` or by the runtime during supervision and recovery procedures.

## `ActorError`

`ActorError` represents a failure encountered while serving an actor. It includes the actor's identity and the underlying cause.
```rust
pub struct ActorError {
    actor_id: ActorId,
    kind: ActorErrorKind,
}
```
This error type is returned in various actor lifecycle operations such as initialization, message handling, checkpointing, and shutdown. It is structured and extensible, allowing the runtime to distinguish between different classes of failure.

### Associated Methods
```rust
impl ActorError {
    /// Constructs a new `ActorError` with the given ID and kind.
    pub(crate) fn new(actor_id: ActorId, kind: ActorErrorKind) -> Self

    /// Returns a cloneable version of this error, discarding error structure
    /// and retaining only the formatted string.
    fn passthrough(&self) -> Self
}
```

## `ActorErrorKind`

```rust
pub enum ActorErrorKind {
    Processing(anyhow::Error),
    Panic(anyhow::Error),
    Init(anyhow::Error),
    Mailbox(MailboxError),
    MailboxSender(MailboxSenderError),
    Checkpoint(CheckpointError),
    MessageLog(MessageLogError),
    IndeterminateState,
    Passthrough(anyhow::Error),
}
```
### Variants

- `Processing`: The actor's `handle()` method returned an error.
- `Panic`: A panic occurred during message handling or actor logic.
- `Init`: Actor initialization failed.
- `Mailbox`: A lower-level mailbox error occurred.
- `MailboxSender`: A lower-level sender error occurred.
- `Checkpoint`: Error during save/load of actor state.
- `MessageLog`: Failure in the underlying message log.
- `IndeterminateState`: The actor reached an invalid or unknown internal state.
- `Passthrough`: A generic error, preserving only the error message.

`Passthrough` is used when a structured error needs to be simplified for cloning or propagation across boundaries.
