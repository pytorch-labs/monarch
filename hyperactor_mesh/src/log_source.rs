/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// TODO: rename this file to logging.rs and delete all non-related code

use std::fmt;
use std::str::FromStr;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor::Context;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor::ProcId;
use hyperactor::RefClient;
use hyperactor::WorldId;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelRx;
use hyperactor::channel::ChannelTransport;
use hyperactor::channel::Rx;
use hyperactor::data::Serialized;
use hyperactor::mailbox;
use hyperactor::mailbox::BoxedMailboxSender;
use hyperactor::mailbox::DialMailboxRouter;
use hyperactor::mailbox::MailboxServer;
use hyperactor::message::Bind;
use hyperactor::message::Bindings;
use hyperactor::message::Unbind;
use hyperactor::proc::Proc;
use hyperactor_state::log_writer::LogMessage;
use hyperactor_state::log_writer::LogMessageClient;
use hyperactor_state::log_writer::LogMessageHandler;
use hyperactor_state::log_writer::OutputTarget;
use hyperactor_state::state_actor::StateActor;
use serde::Deserialize;
use serde::Serialize;

use crate::bootstrap::BOOTSTRAP_LOG_CHANNEL;
use crate::shortuuid::ShortUuid;

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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

    use async_trait::async_trait;
    use hyperactor::channel;
    use hyperactor::channel::ChannelAddr;
    use hyperactor::channel::ChannelTx;
    use hyperactor::channel::Tx;
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

    #[async_trait]
    impl LogHandler for MpscLogHandler {
        async fn handle_log(&self, logs: Vec<GenericStateObject>) -> anyhow::Result<()> {
            self.sender.send(logs).await.unwrap();
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
