/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// TODO: rename this file to logging.rs and delete all non-related code

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorRef;
use hyperactor::Context;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor::RefClient;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelRx;
use hyperactor::channel::ChannelTransport;
use hyperactor::channel::Rx;
use hyperactor::data::Serialized;
use hyperactor::message::Bind;
use hyperactor::message::Bindings;
use hyperactor::message::Unbind;
use hyperactor_state::log_writer::LogMessage;
use hyperactor_state::log_writer::LogMessageClient;
use hyperactor_state::log_writer::LogMessageHandler;
use hyperactor_state::log_writer::OutputTarget;
use serde::Deserialize;
use serde::Serialize;

use crate::bootstrap::BOOTSTRAP_LOG_CHANNEL;

/// Messages that can be sent to the LogWriterActor
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    Named,
    Handler,
    HandleClient,
    RefClient
)]
pub enum LogForwardMessage {
    /// Receive the log from the parent process and forward ti to the client.
    Forward {},

    /// If to stream the log back to the client.
    SetMode { stream_to_client: bool },
}

impl Bind for LogForwardMessage {
    fn bind(&mut self, _bindings: &mut Bindings) -> anyhow::Result<()> {
        Ok(())
    }
}

impl Unbind for LogForwardMessage {
    fn unbind(&self, _bindings: &mut Bindings) -> anyhow::Result<()> {
        Ok(())
    }
}

/// A log forwarder that receives the log from its parent process and forward it back to the client
#[derive(Debug)]
#[hyperactor::export(
    spawn = true,
    handlers = [LogForwardMessage {cast = true}],
)]
pub struct LogForwardActor {
    rx: ChannelRx<LogMessage>,
    logging_client_ref: ActorRef<LogClientActor>,
    stream_to_client: bool,
}

#[async_trait]
impl Actor for LogForwardActor {
    type Params = ActorRef<LogClientActor>;

    async fn new(logging_client_ref: Self::Params) -> Result<Self> {
        let log_channel: ChannelAddr = match std::env::var(BOOTSTRAP_LOG_CHANNEL) {
            Ok(channel) => channel.parse()?,
            Err(err) => {
                tracing::error!(
                    "log forwarder actor failed to read env var {}: {}",
                    BOOTSTRAP_LOG_CHANNEL,
                    err
                );
                // TODO: an empty channel to serve
                ChannelAddr::any(ChannelTransport::Unix)
            }
        };
        tracing::info!(
            "log forwarder {} serve at {}",
            std::process::id(),
            log_channel
        );

        let rx = match channel::serve(log_channel.clone()).await {
            Ok((_, rx)) => rx,
            Err(err) => {
                // This can happen if we are not spanwed on a separate process like local.
                // For local mesh, log streaming anyway is not needed.
                tracing::error!(
                    "log forwarder actor failed to bootstrap on given channel {}: {}",
                    log_channel,
                    err
                );
                channel::serve(ChannelAddr::any(ChannelTransport::Unix))
                    .await?
                    .1
            }
        };
        Ok(Self {
            rx,
            logging_client_ref,
            stream_to_client: false,
        })
    }

    async fn init(&mut self, this: &Instance<Self>) -> Result<(), anyhow::Error> {
        this.self_message_with_delay(LogForwardMessage::Forward {}, Duration::from_secs(0))?;
        Ok(())
    }
}

#[async_trait]
#[hyperactor::forward(LogForwardMessage)]
impl LogForwardMessageHandler for LogForwardActor {
    async fn forward(&mut self, ctx: &Context<Self>) -> Result<(), anyhow::Error> {
        if let Ok(LogMessage::Log {
            hostname,
            pid,
            output_target,
            payload,
        }) = self.rx.recv().await
        {
            if self.stream_to_client {
                self.logging_client_ref
                    .log(ctx, hostname, pid, output_target, payload)
                    .await?;
            }
        }

        // This is not ideal as we are using raw tx/rx.
        ctx.self_message_with_delay(LogForwardMessage::Forward {}, Duration::from_secs(0))?;

        Ok(())
    }

    async fn set_mode(
        &mut self,
        _ctx: &Context<Self>,
        stream_to_client: bool,
    ) -> Result<(), anyhow::Error> {
        self.stream_to_client = stream_to_client;

        Ok(())
    }
}

/// Deserialize a serialized message and split it into UTF-8 lines
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
    spawn = true,
    handlers = [LogMessage],
)]
pub struct LogClientActor {}

#[async_trait]
impl Actor for LogClientActor {
    type Params = ();

    async fn new(_: ()) -> Result<Self, anyhow::Error> {
        Ok(Self {})
    }
}

#[async_trait]
#[hyperactor::forward(LogMessage)]
impl LogMessageHandler for LogClientActor {
    async fn log(
        &mut self,
        _cx: &Context<Self>,
        hostname: String,
        pid: u32,
        output_target: OutputTarget,
        payload: Serialized,
    ) -> Result<(), anyhow::Error> {
        // Deserialize the message and process line by line with UTF-8
        let message_lines = deserialize_message_lines(&payload)?;

        match output_target {
            OutputTarget::Stdout => {
                for line in message_lines {
                    println!("[{} {}] {}", hostname, pid, line);
                }
            }
            OutputTarget::Stderr => {
                for line in message_lines {
                    eprintln!("[{} {}] {}", hostname, pid, line);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use hyperactor::channel;
    use hyperactor::channel::ChannelAddr;
    use hyperactor::channel::ChannelTx;
    use hyperactor::channel::Tx;
    use hyperactor::id;
    use hyperactor::mailbox;
    use hyperactor::mailbox::BoxedMailboxSender;
    use hyperactor::mailbox::DialMailboxRouter;
    use hyperactor::mailbox::MailboxServer;
    use hyperactor::proc::Proc;

    use super::*;

    #[tokio::test]
    async fn test_forwarding_log_to_client() {
        // Setup the basics
        let router = DialMailboxRouter::new();
        let (proc_addr, client_rx) = channel::serve(ChannelAddr::any(ChannelTransport::Unix))
            .await
            .unwrap();
        let proc = Proc::new(id!(client[0]), BoxedMailboxSender::new(router.clone()));
        proc.clone()
            .serve(client_rx, mailbox::monitored_return_handle());
        router.bind(id!(client[0]).into(), proc_addr.clone());
        let client = proc.attach("client").unwrap();

        // Spin up both the forwarder and the client
        let log_channel = ChannelAddr::any(ChannelTransport::Unix);
        // SAFETY: Unit test
        unsafe {
            std::env::set_var(BOOTSTRAP_LOG_CHANNEL, log_channel.to_string());
        }
        let log_client: ActorRef<LogClientActor> =
            proc.spawn("log_client", ()).await.unwrap().bind();
        let log_forwarder: ActorRef<LogForwardActor> = proc
            .spawn("log_forwarder", log_client)
            .await
            .unwrap()
            .bind();

        // Write some logs that will not be streamed
        let tx: ChannelTx<LogMessage> = channel::dial(log_channel).unwrap();
        tx.post(LogMessage::Log {
            hostname: "my_host".into(),
            pid: 1,
            output_target: OutputTarget::Stderr,
            payload: Serialized::serialize_anon(&"will not stream".to_string()).unwrap(),
        });

        // Turn on streaming
        log_forwarder.set_mode(&client, true).await.unwrap();
        tx.post(LogMessage::Log {
            hostname: "my_host".into(),
            pid: 1,
            output_target: OutputTarget::Stderr,
            payload: Serialized::serialize_anon(&"will stream".to_string()).unwrap(),
        });

        // TODO: it is hard to test out anything meaningful here as the client flushes to stdout.
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
