/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::Context;
use hyperactor::Handler;
use hyperactor::Named;
use hyperactor_macros::HandleClient;
use hyperactor_macros::RefClient;
use serde::Deserialize;
use serde::Serialize;

use crate::object::GenericStateObject;

/// A trait for handling logs received by the client actor.
#[async_trait]
pub trait LogHandler: Sync + Send + std::fmt::Debug + 'static {
    /// Handle the logs received by the client actor.
    async fn handle_log(&self, logs: Vec<GenericStateObject>) -> Result<()>;
}

/// A client to interact with the state actor.
#[derive(Debug)]
#[hyperactor::export(
    handlers = [ClientMessage],
)]
pub struct ClientActor {
    // Use boxed log handler to erase types.
    // This is needed because the client actor ref needs to be sent over the wire and used by the state actor.
    log_handler: Box<dyn LogHandler>,
}

/// Endpoints for the client actor.
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub enum ClientMessage {
    /// Push a batch of logs to the logs buffer
    PushLogs { logs: Vec<GenericStateObject> },
}

pub struct ClientActorParams {
    pub log_handler: Box<dyn LogHandler>,
}

#[async_trait]
impl Actor for ClientActor {
    type Params = ClientActorParams;

    async fn new(
        ClientActorParams { log_handler }: ClientActorParams,
    ) -> Result<Self, anyhow::Error> {
        Ok(Self { log_handler })
    }
}

#[async_trait]
#[hyperactor::forward(ClientMessage)]
impl ClientMessageHandler for ClientActor {
    async fn push_logs(
        &mut self,
        _cx: &Context<Self>,
        logs: Vec<GenericStateObject>,
    ) -> Result<(), anyhow::Error> {
        self.log_handler.handle_log(logs).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use hyperactor::ActorRef;
    use hyperactor::channel;
    use hyperactor::channel::ChannelAddr;
    use hyperactor::clock::Clock;
    use tokio::sync::mpsc::Sender;

    use super::*;
    use crate::create_remote_client;
    use crate::test_utils::log_items;
    use crate::test_utils::spawn_actor;

    /// A log handler that flushes GenericStateObject to a mpsc channel.
    #[derive(Debug)]
    struct MpscLogHandler {
        sender: Sender<Vec<GenericStateObject>>,
    }

    #[async_trait]
    impl LogHandler for MpscLogHandler {
        async fn handle_log(&self, logs: Vec<GenericStateObject>) -> Result<()> {
            self.sender.send(logs).await.unwrap();
            Ok(())
        }
    }

    #[tracing_test::traced_test]
    #[tokio::test]
    async fn test_client_basics() {
        let client_actor_addr = ChannelAddr::any(channel::ChannelTransport::Unix);
        let (sender, mut receiver) = tokio::sync::mpsc::channel::<Vec<GenericStateObject>>(10);
        let log_handler = Box::new(MpscLogHandler { sender });
        let params = ClientActorParams { log_handler };
        let client_proc_id =
            hyperactor::reference::ProcId(hyperactor::WorldId("client_server".to_string()), 0);
        let (client_actor_addr, client_actor_handle, _client_mailbox) = spawn_actor::<ClientActor>(
            client_actor_addr.clone(),
            client_proc_id,
            "state_client",
            params,
        )
        .await
        .unwrap();
        let client_actor_ref: ActorRef<ClientActor> = client_actor_handle.bind();

        let (_client_proc, remote_client) = create_remote_client(client_actor_addr).await.unwrap();

        let log_items_0_10 = log_items(0, 10);
        client_actor_ref
            .push_logs(&remote_client, log_items_0_10.clone())
            .await
            .unwrap();

        // Collect received messages with timeout
        let clock = hyperactor::clock::ClockKind::default();
        let fetched_logs = clock
            .timeout(Duration::from_secs(1), receiver.recv())
            .await
            .expect("timed out waiting for message")
            .expect("channel closed unexpectedly");

        // Verify we received all expected logs
        assert_eq!(fetched_logs.len(), 10);
        assert_eq!(fetched_logs, log_items_0_10);

        // Now test that no extra message is waiting
        let extra = clock
            .timeout(Duration::from_millis(100), receiver.recv())
            .await;
        assert!(extra.is_err(), "expected no more messages");
    }
}
