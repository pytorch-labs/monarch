/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::str::FromStr;

use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor::ProcId;
use hyperactor::WorldId;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelTransport;
use hyperactor::mailbox;
use hyperactor::mailbox::BoxedMailboxSender;
use hyperactor::mailbox::DialMailboxRouter;
use hyperactor::mailbox::MailboxServer;
use hyperactor::proc::Proc;
use hyperactor_state::state_actor::StateActor;

use crate::shortuuid::ShortUuid;

/// The source of the log so that the remote process can stream stdout and stderr to.
/// A log source is allocation specific. Each allocator can decide how to stream the logs back.
/// It holds a reference or the lifecycle of a state actor that collects all the logs from processes.
#[derive(Clone, Debug)]
pub struct LogSource {
    // Optionally hold the lifecycle of the state actor
    _state_proc: Option<Proc>,
    state_proc_addr: ChannelAddr,
    state_actor: ActorRef<StateActor>,
}

#[derive(Clone, Debug)]
pub struct StateServerInfo {
    pub state_proc_addr: ChannelAddr,
    pub state_actor_id: ActorId,
}

impl fmt::Display for StateServerInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{},{}", self.state_proc_addr, self.state_actor_id)
    }
}

impl FromStr for StateServerInfo {
    type Err = anyhow::Error;

    fn from_str(state_server_info: &str) -> Result<Self, Self::Err> {
        match state_server_info.split_once(",") {
            Some((addr, id)) => {
                let state_proc_addr: ChannelAddr = addr.parse()?;
                let state_actor_id: ActorId = id.parse()?;
                Ok(StateServerInfo {
                    state_proc_addr,
                    state_actor_id,
                })
            }
            _ => Err(anyhow::anyhow!(
                "unrecognized state server info: {state_server_info}"
            )),
        }
    }
}

impl LogSource {
    /// Spin up the state actor locally to receive the remote logs.
    pub async fn new_with_local_actor() -> Result<Self, anyhow::Error> {
        let router = DialMailboxRouter::new();
        let state_proc_id = ProcId(WorldId(format!("local_state_{}", ShortUuid::generate())), 0);
        let (state_proc_addr, state_rx) =
            channel::serve(ChannelAddr::any(ChannelTransport::Unix)).await?;
        let state_proc = Proc::new(
            state_proc_id.clone(),
            BoxedMailboxSender::new(router.clone()),
        );
        state_proc
            .clone()
            .serve(state_rx, mailbox::monitored_return_handle());
        router.bind(state_proc_id.clone().into(), state_proc_addr.clone());
        let state_actor = state_proc.spawn::<StateActor>("state", ()).await?.bind();

        Ok(Self {
            _state_proc: Some(state_proc),
            state_proc_addr,
            state_actor,
        })
    }

    /// Connect to an existing state actor to receive the remote logs.
    /// The lifecycle of the state actor should be maintained by the allocator who creates it.
    pub fn new_with_remote_actor(state_proc_id: ActorId, state_proc_addr: ChannelAddr) -> Self {
        Self {
            _state_proc: None,
            state_proc_addr,
            state_actor: ActorRef::attest(state_proc_id),
        }
    }

    pub fn server_info(&self) -> StateServerInfo {
        StateServerInfo {
            state_proc_addr: self.state_proc_addr.clone(),
            state_actor_id: self.state_actor.actor_id().clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use std::time::Duration;

    use hyperactor::channel;
    use hyperactor::channel::ChannelAddr;
    use hyperactor::clock::Clock;
    use hyperactor::id;
    use hyperactor_state::client::ClientActor;
    use hyperactor_state::client::ClientActorParams;
    use hyperactor_state::client::LogHandler;
    use hyperactor_state::object::GenericStateObject;
    use hyperactor_state::state_actor::StateMessageClient;
    use hyperactor_state::test_utils::log_items;
    use tokio::sync::mpsc::Sender;

    use super::*;

    #[test]
    fn test_state_server_info_serialization() {
        let addr = ChannelAddr::any(channel::ChannelTransport::Unix);
        let actor_id: ActorId = id!(test_actor[42].actor[0]);

        let info = StateServerInfo {
            state_proc_addr: addr.clone(),
            state_actor_id: actor_id.clone(),
        };

        // Test Display implementation
        let serialized = format!("{}", info);
        assert!(serialized.contains(","));

        // Test FromStr implementation
        let deserialized = StateServerInfo::from_str(&serialized).unwrap();
        assert_eq!(
            info.state_proc_addr.to_string(),
            deserialized.state_proc_addr.to_string()
        );
        assert_eq!(
            info.state_actor_id.to_string(),
            deserialized.state_actor_id.to_string()
        );
    }

    #[derive(Debug)]
    struct MpscLogHandler {
        sender: Sender<Vec<GenericStateObject>>,
    }

    impl LogHandler for MpscLogHandler {
        fn handle_log(&self, logs: Vec<GenericStateObject>) -> anyhow::Result<()> {
            let sender = self.sender.clone();
            tokio::spawn(async move {
                sender.send(logs).await.unwrap();
            });
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_state_server_pushing_logs() {
        // Spin up a new state actor
        let log_source = LogSource::new_with_local_actor().await.unwrap();

        // Setup the client and connect it to the state actor
        let router = DialMailboxRouter::new();
        let (client_proc_addr, client_rx) =
            channel::serve(ChannelAddr::any(ChannelTransport::Unix))
                .await
                .unwrap();
        let client_proc = Proc::new(id!(client[0]), BoxedMailboxSender::new(router.clone()));
        client_proc
            .clone()
            .serve(client_rx, mailbox::monitored_return_handle());
        router.bind(id!(client[0]).into(), client_proc_addr.clone());
        router.bind(
            log_source.server_info().state_actor_id.clone().into(),
            log_source.server_info().state_proc_addr.clone(),
        );
        let client = client_proc.attach("client").unwrap();

        // Spin up the client logging actor and subscribe to the state actor
        let (sender, mut receiver) = tokio::sync::mpsc::channel::<Vec<GenericStateObject>>(20);
        let log_handler = Box::new(MpscLogHandler { sender });
        let params = ClientActorParams { log_handler };

        let client_logging_actor: ActorRef<ClientActor> = client_proc
            .spawn::<ClientActor>("logging_client", params)
            .await
            .unwrap()
            .bind();
        let state_actor_ref: ActorRef<StateActor> =
            ActorRef::attest(log_source.server_info().state_actor_id.clone());
        state_actor_ref
            .subscribe_logs(
                &client,
                client_proc_addr.clone(),
                client_logging_actor.clone(),
            )
            .await
            .unwrap();

        // Write some logs
        state_actor_ref
            .push_logs(&client, log_items(0, 10))
            .await
            .unwrap();

        // Collect received messages with timeout
        let fetched_logs = client_proc
            .clock()
            .timeout(Duration::from_secs(1), receiver.recv())
            .await
            .expect("timed out waiting for message")
            .expect("channel closed unexpectedly");

        // Verify we received all expected logs
        assert_eq!(fetched_logs.len(), 10);
        assert_eq!(fetched_logs, log_items(0, 10));
    }
}
