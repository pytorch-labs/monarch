/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The mesh agent actor manages procs in ProcMeshes.

use std::collections::HashMap;
use std::mem::replace;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;

use async_trait::async_trait;
use enum_as_inner::EnumAsInner;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::ActorId;
use hyperactor::Context;
use hyperactor::Data;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor::PortHandle;
use hyperactor::PortRef;
use hyperactor::ProcId;
use hyperactor::RefClient;
use hyperactor::actor::remote::Remote;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::mailbox::BoxedMailboxSender;
use hyperactor::mailbox::DeliveryError;
use hyperactor::mailbox::DialMailboxRouter;
use hyperactor::mailbox::IntoBoxedMailboxSender;
use hyperactor::mailbox::MailboxClient;
use hyperactor::mailbox::MailboxSender;
use hyperactor::mailbox::MessageEnvelope;
use hyperactor::mailbox::Undeliverable;
use hyperactor::mailbox::UndeliverableMessageError;
use hyperactor::proc::Proc;
use hyperactor::supervision::ActorSupervisionEvent;
use serde::Deserialize;
use serde::Serialize;

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    Handler,
    HandleClient,
    RefClient,
    Named
)]
pub(crate) enum MeshAgentMessage {
    /// Configure the proc in the mesh.
    Configure {
        /// The rank of this proc in the mesh.
        rank: usize,
        /// The forwarder to send messages to unknown destinations.
        forwarder: ChannelAddr,
        /// The supervisor port to which the agent should report supervision events.
        supervisor: PortRef<ActorSupervisionEvent>,
        /// An address book to use for direct dialing.
        address_book: HashMap<ProcId, ChannelAddr>,
        /// The agent should write its rank to this port when it successfully
        /// configured.
        configured: PortRef<usize>,
    },

    /// Spawn an actor on the proc to the provided name.
    Gspawn {
        /// registered actor type
        actor_type: String,
        /// spawned actor name
        actor_name: String,
        /// serialized parameters
        params_data: Data,
        /// reply port; the proc should send its rank to indicated a spawned actor
        status_port: PortRef<(usize, ActorId)>,
    },
}

/// A mesh agent is responsible for managing procs in a [`ProcMesh`].
#[derive(Debug)]
#[hyperactor::export(handlers=[MeshAgentMessage])]
pub struct MeshAgent {
    proc: Proc,
    remote: Remote,
    sender: ReconfigurableMailboxSender,
    rank: Option<usize>,
    supervisor: Option<PortRef<ActorSupervisionEvent>>,
}

impl MeshAgent {
    #[hyperactor::instrument]
    pub(crate) async fn bootstrap(
        proc_id: ProcId,
    ) -> Result<(Proc, ActorHandle<Self>), anyhow::Error> {
        let sender = ReconfigurableMailboxSender::new();
        let proc = Proc::new(proc_id, BoxedMailboxSender::new(sender.clone()));

        let agent = MeshAgent {
            proc: proc.clone(),
            remote: Remote::collect(),
            sender,
            rank: None,       // not yet assigned
            supervisor: None, // not yet assigned
        };
        let handle = proc.spawn::<Self>("mesh", agent).await?;
        tracing::info!("bootstrap_end");
        Ok((proc, handle))
    }
}

#[async_trait]
impl Actor for MeshAgent {
    type Params = Self;

    async fn new(params: Self::Params) -> Result<Self, anyhow::Error> {
        Ok(params)
    }

    async fn init(&mut self, this: &Instance<Self>) -> Result<(), anyhow::Error> {
        self.proc.set_supervision_coordinator(this.port())?;
        Ok(())
    }

    // This is an override of the default actor behavior.
    async fn handle_undeliverable_message(
        &mut self,
        this: &Instance<Self>,
        undelivered: Undeliverable<MessageEnvelope>,
    ) -> Result<(), anyhow::Error> {
        let Undeliverable(ref envelope) = undelivered;
        tracing::debug!("took charge of a message not delivered: {}", envelope);

        let sender = envelope.sender().clone();
        if this.self_id() == &sender {
            anyhow::bail!(UndeliverableMessageError::delivery_failure(envelope));
        }

        let mut envelope = envelope.clone();
        let return_port = PortRef::attest_message_port(&sender);
        return_port.send(this, undelivered).map_err(|err| {
            envelope.try_set_error(DeliveryError::BrokenLink(format!("send failure: {err}")));
            UndeliverableMessageError::return_failure(&envelope)
        })?;

        Ok(())
    }
}

#[async_trait]
#[hyperactor::forward(MeshAgentMessage)]
impl MeshAgentMessageHandler for MeshAgent {
    async fn configure(
        &mut self,
        this: &Context<Self>,
        rank: usize,
        forwarder: ChannelAddr,
        supervisor: PortRef<ActorSupervisionEvent>,
        address_book: HashMap<ProcId, ChannelAddr>,
        configured: PortRef<usize>,
    ) -> Result<(), anyhow::Error> {
        // Set the supervisor first so that we can handle supervison events that might
        // occur from configuration failures. Though we should instead report these directly
        // for better ergonomics in the allocator.
        self.supervisor = Some(supervisor);
        let client = MailboxClient::new(channel::dial(forwarder)?);
        let router = DialMailboxRouter::new_with_default(client.into_boxed());
        for (proc_id, addr) in address_book {
            router.bind(proc_id.into(), addr);
        }

        if self.sender.configure(router.into_boxed()) {
            self.rank = Some(rank);
            configured.send(this, rank)?;
        } else {
            tracing::error!("tried to reconfigure mesh agent");
        }
        Ok(())
    }

    async fn gspawn(
        &mut self,
        this: &Context<Self>,
        actor_type: String,
        actor_name: String,
        params_data: Data,
        status_port: PortRef<(usize, ActorId)>,
    ) -> Result<(), anyhow::Error> {
        let actor_id = self
            .remote
            .gspawn(&self.proc, &actor_type, &actor_name, params_data)
            .await?;
        let rank = self
            .rank
            .ok_or_else(|| anyhow::anyhow!("tried to spawn on unconfigured proc"))?;
        status_port.send(this, (rank, actor_id))?;
        Ok(())
    }
}

#[async_trait]
impl Handler<ActorSupervisionEvent> for MeshAgent {
    async fn handle(
        &mut self,
        this: &Context<Self>,
        event: ActorSupervisionEvent,
    ) -> anyhow::Result<()> {
        if let Some(supervisor) = &self.supervisor {
            supervisor.send(this, event)?;
        } else {
            tracing::error!(
                "proc {}: could not propagate supervision event {:?}: crashing",
                this.self_id().proc_id(),
                event
            );

            // We should have a custom "crash" function here, so that this works
            // in testing of the LocalAllocator, etc.
            std::process::exit(1);
        }
        Ok(())
    }
}

/// A mailbox sender that initially queues messages, and then relays them to
/// an underlying sender once configured.
#[derive(Clone)]
pub(crate) struct ReconfigurableMailboxSender {
    state: Arc<RwLock<ReconfigurableMailboxSenderState>>,
}

impl std::fmt::Debug for ReconfigurableMailboxSender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Not super helpful, but we definitely don't wan to acquire any locks
        // in a Debug formatter.
        f.debug_struct("ReconfigurableMailboxSender").finish()
    }
}

type Post = (MessageEnvelope, PortHandle<Undeliverable<MessageEnvelope>>);

#[derive(EnumAsInner, Debug)]
enum ReconfigurableMailboxSenderState {
    Queueing(Mutex<Vec<Post>>),
    Configured(BoxedMailboxSender),
}

impl ReconfigurableMailboxSender {
    pub(crate) fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(ReconfigurableMailboxSenderState::Queueing(
                Mutex::new(Vec::new()),
            ))),
        }
    }

    /// Configure this mailbox with the provided sender. This will first
    /// enqueue any pending messages onto the sender; future messages are
    /// posted directly to the configured sender.
    pub(crate) fn configure(&self, sender: BoxedMailboxSender) -> bool {
        let mut state = self.state.write().unwrap();
        if state.is_configured() {
            return false;
        }

        let queued = replace(
            &mut *state,
            ReconfigurableMailboxSenderState::Configured(sender.clone()),
        );

        for (envelope, return_handle) in queued.into_queueing().unwrap().into_inner().unwrap() {
            sender.post(envelope, return_handle);
        }
        *state = ReconfigurableMailboxSenderState::Configured(sender);
        true
    }
}

impl MailboxSender for ReconfigurableMailboxSender {
    fn post(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        match *self.state.read().unwrap() {
            ReconfigurableMailboxSenderState::Queueing(ref queue) => {
                queue.lock().unwrap().push((envelope, return_handle));
            }
            ReconfigurableMailboxSenderState::Configured(ref sender) => {
                sender.post(envelope, return_handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;

    use hyperactor::attrs::Attrs;
    use hyperactor::id;
    use hyperactor::mailbox::BoxedMailboxSender;
    use hyperactor::mailbox::Mailbox;
    use hyperactor::mailbox::MailboxSender;
    use hyperactor::mailbox::MessageEnvelope;
    use hyperactor::mailbox::PortHandle;
    use hyperactor::mailbox::Undeliverable;

    use super::*;

    #[derive(Debug, Clone)]
    struct QueueingMailboxSender {
        messages: Arc<Mutex<Vec<MessageEnvelope>>>,
    }

    impl QueueingMailboxSender {
        fn new() -> Self {
            Self {
                messages: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn get_messages(&self) -> Vec<MessageEnvelope> {
            self.messages.lock().unwrap().clone()
        }
    }

    impl MailboxSender for QueueingMailboxSender {
        fn post(
            &self,
            envelope: MessageEnvelope,
            _return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
        ) {
            self.messages.lock().unwrap().push(envelope);
        }
    }

    // Helper function to create a test message envelope
    fn envelope(data: u64) -> MessageEnvelope {
        MessageEnvelope::serialize(
            id!(world[0].sender),
            id!(world[0].receiver[0][1]),
            &data,
            Attrs::new(),
        )
        .unwrap()
    }

    fn return_handle() -> PortHandle<Undeliverable<MessageEnvelope>> {
        let mbox = Mailbox::new_detached(id!(test[0].test));
        let (port, _receiver) = mbox.open_port::<Undeliverable<MessageEnvelope>>();
        port
    }

    #[test]
    fn test_queueing_before_configure() {
        let sender = ReconfigurableMailboxSender::new();

        let test_sender = QueueingMailboxSender::new();
        let boxed_sender = BoxedMailboxSender::new(test_sender.clone());

        let return_handle = return_handle();
        sender.post(envelope(1), return_handle.clone());
        sender.post(envelope(2), return_handle.clone());

        assert_eq!(test_sender.get_messages().len(), 0);

        sender.configure(boxed_sender);

        let messages = test_sender.get_messages();
        assert_eq!(messages.len(), 2);

        assert_eq!(messages[0].deserialized::<u64>().unwrap(), 1);
        assert_eq!(messages[1].deserialized::<u64>().unwrap(), 2);
    }

    #[test]
    fn test_direct_delivery_after_configure() {
        // Create a ReconfigurableMailboxSender
        let sender = ReconfigurableMailboxSender::new();

        let test_sender = QueueingMailboxSender::new();
        let boxed_sender = BoxedMailboxSender::new(test_sender.clone());
        sender.configure(boxed_sender);

        let return_handle = return_handle();
        sender.post(envelope(3), return_handle.clone());
        sender.post(envelope(4), return_handle.clone());

        let messages = test_sender.get_messages();
        assert_eq!(messages.len(), 2);

        assert_eq!(messages[0].deserialized::<u64>().unwrap(), 3);
        assert_eq!(messages[1].deserialized::<u64>().unwrap(), 4);
    }

    #[test]
    fn test_multiple_configurations() {
        let sender = ReconfigurableMailboxSender::new();
        let boxed_sender = BoxedMailboxSender::new(QueueingMailboxSender::new());

        assert!(sender.configure(boxed_sender.clone()));
        assert!(!sender.configure(boxed_sender));
    }

    #[test]
    fn test_mixed_queueing_and_direct_delivery() {
        let sender = ReconfigurableMailboxSender::new();

        let test_sender = QueueingMailboxSender::new();
        let boxed_sender = BoxedMailboxSender::new(test_sender.clone());

        let return_handle = return_handle();
        sender.post(envelope(5), return_handle.clone());
        sender.post(envelope(6), return_handle.clone());

        sender.configure(boxed_sender);

        sender.post(envelope(7), return_handle.clone());
        sender.post(envelope(8), return_handle.clone());

        let messages = test_sender.get_messages();
        assert_eq!(messages.len(), 4);

        assert_eq!(messages[0].deserialized::<u64>().unwrap(), 5);
        assert_eq!(messages[1].deserialized::<u64>().unwrap(), 6);
        assert_eq!(messages[2].deserialized::<u64>().unwrap(), 7);
        assert_eq!(messages[3].deserialized::<u64>().unwrap(), 8);
    }
}
