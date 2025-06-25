/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Mailbox;
use hyperactor::Named;
use hyperactor::PortRef;
use hyperactor::RefClient;
use hyperactor::channel::ChannelAddr;
use hyperactor::proc::Proc;
use serde::Deserialize;
use serde::Serialize;

use crate::client::ClientMessage;
use crate::create_remote_client;
use crate::object::GenericStateObject;

/// A state actor which serves as a centralized store for state.
#[derive(Debug)]
#[hyperactor::export(
    handlers = [StateMessage],
)]
pub struct StateActor {
    // Using PortRef instead of ActorRef<ClientActor> allows any actor to become a subscriber
    log_subscribers: HashMap<PortRef<ClientMessage>, (Proc, Mailbox)>,
}

/// Endpoints for the state actor.
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub enum StateMessage {
    /// Send a batch of logs to the state actor.
    SetLogs { logs: Vec<GenericStateObject> },
    /// Log subscription messages from client. This message should be sent by any actor to
    /// inform the state actor to start sending logs to the client.
    SubscribeLogs {
        addr: ChannelAddr,
        subscriber_port: PortRef<ClientMessage>,
    },
    /// Unsubscribe from logs. This message should be sent by any actor to inform
    /// the state actor to stop sending logs to the client.
    UnsubscribeLogs {
        subscriber_port: PortRef<ClientMessage>,
    },
}

#[async_trait]
impl Actor for StateActor {
    type Params = ();

    async fn new(_params: ()) -> Result<Self, anyhow::Error> {
        Ok(Self {
            log_subscribers: HashMap::new(),
        })
    }
}

#[async_trait]
#[hyperactor::forward(StateMessage)]
impl StateMessageHandler for StateActor {
    async fn set_logs(
        &mut self,
        _this: &Instance<Self>,
        logs: Vec<GenericStateObject>,
    ) -> Result<(), anyhow::Error> {
        for (port, (_, remote_client)) in self.log_subscribers.iter() {
            // Send logs message directly to the port
            port.send(remote_client, ClientMessage::Logs { logs: logs.clone() })?;
        }
        Ok(())
    }

    async fn subscribe_logs(
        &mut self,
        _this: &Instance<Self>,
        addr: ChannelAddr,
        subscriber_port: PortRef<ClientMessage>,
    ) -> Result<(), anyhow::Error> {
        tracing::info!("Subscribing client with port at {} to logs", &addr);
        self.log_subscribers
            .insert(subscriber_port, create_remote_client(addr).await?);
        Ok(())
    }

    async fn unsubscribe_logs(
        &mut self,
        _this: &Instance<Self>,
        subscriber_port: PortRef<ClientMessage>,
    ) -> Result<(), anyhow::Error> {
        tracing::info!("Unsubscribing client from logs");
        self.log_subscribers.remove(&subscriber_port);
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use std::time::Duration;

    use hyperactor::ActorRef;
    use hyperactor::channel;

    use super::*;
    use crate::client::ClientActor;
    use crate::client::ClientActorParams;
    use crate::create_remote_client;
    use crate::spawn_actor;
    use crate::test_utils::log_items;

    #[tokio::test]
    async fn test_subscribe_logs() {
        let state_actor_addr = ChannelAddr::any(channel::ChannelTransport::Unix);
        let state_proc_id =
            hyperactor::reference::ProcId(hyperactor::WorldId("state_server".to_string()), 0);
        let (state_actor_addr, state_actor_handle) =
            spawn_actor::<StateActor>(state_actor_addr.clone(), state_proc_id, "state", ())
                .await
                .unwrap();
        let state_actor_ref: ActorRef<StateActor> = state_actor_handle.bind();

        let client_actor_addr = ChannelAddr::any(channel::ChannelTransport::Unix);
        let (sender, mut receiver) = tokio::sync::mpsc::channel::<GenericStateObject>(20);
        let params = ClientActorParams { sender };
        let client_proc_id =
            hyperactor::reference::ProcId(hyperactor::WorldId("client_server".to_string()), 0);
        let (client_actor_addr, client_actor_handle) = spawn_actor::<ClientActor>(
            client_actor_addr.clone(),
            client_proc_id,
            "state_client",
            params,
        )
        .await
        .unwrap();
        let client_actor_ref: ActorRef<ClientActor> = client_actor_handle.bind();

        let (_proc, remote_client) = create_remote_client(state_actor_addr).await.unwrap();

        // 1. Subscribe to logs using the client actor's port
        // Convert the client actor reference to a PortRef<ClientMessage>
        let client_port =
            PortRef::<ClientMessage>::attest_message_port(client_actor_ref.actor_id());
        state_actor_ref
            .subscribe_logs(
                &remote_client,
                client_actor_addr.clone(),
                client_port.clone(),
            )
            .await
            .unwrap();

        // 2. Send logs and verify they are received
        state_actor_ref
            .set_logs(&remote_client, log_items(0, 10))
            .await
            .unwrap();

        // Collect received messages with timeout
        let mut fetched_logs = vec![];
        for _ in 0..10 {
            // Timeout prevents hanging if a message is missing
            let log = tokio::time::timeout(Duration::from_secs(1), receiver.recv())
                .await
                .expect("timed out waiting for message")
                .expect("channel closed unexpectedly");

            fetched_logs.push(log);
        }

        // Verify we received all expected logs
        assert_eq!(fetched_logs.len(), 10);
        assert_eq!(fetched_logs, log_items(0, 10));

        // Now test that no extra message is waiting
        let extra = tokio::time::timeout(Duration::from_millis(100), receiver.recv()).await;
        assert!(extra.is_err(), "expected no more messages");

        // 3. Unsubscribe from logs
        state_actor_ref
            .unsubscribe_logs(&remote_client, client_port.clone())
            .await
            .unwrap();

        // 4. Send more logs and verify they are not received
        state_actor_ref
            .set_logs(&remote_client, log_items(10, 20))
            .await
            .unwrap();

        // Verify no messages are received after unsubscribing
        let no_logs = tokio::time::timeout(Duration::from_millis(500), receiver.recv()).await;
        assert!(no_logs.is_err(), "expected no messages after unsubscribing");

        // 5. Subscribe again
        state_actor_ref
            .subscribe_logs(&remote_client, client_actor_addr, client_port)
            .await
            .unwrap();

        // 6. Send logs and verify they are received again
        state_actor_ref
            .set_logs(&remote_client, log_items(20, 30))
            .await
            .unwrap();

        // Collect received messages with timeout
        let mut fetched_logs_after_resubscribe = vec![];
        for _ in 0..10 {
            // Timeout prevents hanging if a message is missing
            let log = tokio::time::timeout(Duration::from_secs(1), receiver.recv())
                .await
                .expect("timed out waiting for message after resubscribing")
                .expect("channel closed unexpectedly");

            fetched_logs_after_resubscribe.push(log);
        }

        // Verify we received all expected logs after resubscribing
        assert_eq!(fetched_logs_after_resubscribe.len(), 10);
        assert_eq!(fetched_logs_after_resubscribe, log_items(20, 30));
    }
}
