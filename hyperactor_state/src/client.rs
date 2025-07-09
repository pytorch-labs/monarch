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
use crate::object::Kind;
use crate::object::LogSpec;
use crate::object::LogState;
use crate::object::Name;
use crate::object::StateObject;

pub trait LogHandler: Sync + Send + std::fmt::Debug + 'static {
    // we cannot call it handle here as it conflicts with hyperactor macro
    fn handle_log(&self, logs: Vec<GenericStateObject>) -> Result<()>;
}

/// A log handler that flushes GenericStateObject to stdout.
#[derive(Debug)]
pub struct StdlogHandler;

impl LogHandler for StdlogHandler {
    fn handle_log(&self, logs: Vec<GenericStateObject>) -> Result<()> {
        for log in logs {
            let metadata = log.metadata();
            let deserialized_data: StateObject<LogSpec, LogState> = log.data().deserialized()?;

            // Deserialize the message and process line by line with UTF-8
            let message_lines = deserialize_message_lines(&deserialized_data.state.message)?;

            // TODO: @lky D77377307 do not use raw string to distinguish between stdout and stderr
            if metadata.kind != Kind::Log {
                continue;
            }
            match &metadata.name {
                Name::StdoutLog((hostname, pid)) => {
                    for line in message_lines {
                        // TODO: @lky hostname and pid should only be printed for non-aggregated logs. =
                        // For aggregated logs, we should leave as is for better aggregation.
                        println!("[{} {}] {}", hostname, pid, line);
                    }
                }
                Name::StderrLog((hostname, pid)) => {
                    for line in message_lines {
                        eprintln!("[{} {}] {}", hostname, pid, line);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Deserialize a Serialized message and split it into UTF-8 lines
fn deserialize_message_lines(
    serialized_message: &hyperactor::data::Serialized,
) -> Result<Vec<String>> {
    // Try to deserialize as String first
    if let Ok(message_str) = serialized_message.deserialized::<String>() {
        return Ok(message_str.lines().map(|s| s.to_string()).collect());
    }

    // If that fails, try to deserialize as Vec<u8> and convert to UTF-8
    if let Ok(message_bytes) = serialized_message.deserialized::<Vec<u8>>() {
        let message_str = String::from_utf8(message_bytes)?;
        return Ok(message_str.lines().map(|s| s.to_string()).collect());
    }

    // If both fail, return an error
    anyhow::bail!("Failed to deserialize message as either String or Vec<u8>")
}

/// A client to interact with the state actor.
#[derive(Debug)]
#[hyperactor::export(
    handlers = [ClientMessage],
)]
pub struct ClientActor {
    // TODO: extend hyperactor macro to support a generic to avoid using Box here.
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
        self.log_handler.handle_log(logs)?;
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
    use hyperactor::data::Serialized;
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

    impl LogHandler for MpscLogHandler {
        fn handle_log(&self, logs: Vec<GenericStateObject>) -> Result<()> {
            let sender = self.sender.clone();
            tokio::spawn(async move {
                sender.send(logs).await.unwrap();
            });
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

    #[test]
    fn test_deserialize_message_lines_string() {
        // Test deserializing a String message with multiple lines
        let message = "Line 1\nLine 2\nLine 3".to_string();
        let serialized = Serialized::serialize_anon(&message).unwrap();

        let result = deserialize_message_lines(&serialized).unwrap();

        assert_eq!(result, vec!["Line 1", "Line 2", "Line 3"]);

        // Test deserializing a Vec<u8> message with UTF-8 content
        let message_bytes = "Hello\nWorld\nUTF-8 \u{1F980}".as_bytes().to_vec();
        let serialized = Serialized::serialize_anon(&message_bytes).unwrap();

        let result = deserialize_message_lines(&serialized).unwrap();

        assert_eq!(result, vec!["Hello", "World", "UTF-8 \u{1F980}"]);

        // Test deserializing a single line message
        let message = "Single line message".to_string();
        let serialized = Serialized::serialize_anon(&message).unwrap();

        let result = deserialize_message_lines(&serialized).unwrap();

        assert_eq!(result, vec!["Single line message"]);

        // Test deserializing an empty lines
        let message = "\n\n".to_string();
        let serialized = Serialized::serialize_anon(&message).unwrap();

        let result = deserialize_message_lines(&serialized).unwrap();

        assert_eq!(result, vec!["", ""]);

        // Test error handling for invalid UTF-8 bytes
        let invalid_utf8_bytes = vec![0xFF, 0xFE, 0xFD]; // Invalid UTF-8 sequence
        let serialized = Serialized::serialize_anon(&invalid_utf8_bytes).unwrap();

        let result = deserialize_message_lines(&serialized);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid utf-8"));
    }
}
