/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::pin::Pin;
use std::sync::atomic::AtomicU64;
use std::task::Context;
use std::task::Poll;

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorRef;
use hyperactor::Context as HyperactorContext;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Mailbox;
use hyperactor::Named;
use hyperactor::RefClient;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::data::Serialized;
use hyperactor::id;
use hyperactor::mailbox::BoxedMailboxSender;
use hyperactor::mailbox::DialMailboxRouter;
use hyperactor::proc::Proc;
use serde::Deserialize;
use serde::Serialize;
use tokio::io;

use crate::object::GenericStateObject;
use crate::object::Kind;
use crate::object::LogSpec;
use crate::object::LogState;
use crate::object::Name;
use crate::object::StateMetadata;
use crate::object::StateObject;
use crate::state_actor::StateActor;
use crate::state_actor::StateMessage;

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
pub enum LogWriterMessage {
    /// Set the state actor to send logs to
    SetStateActor {
        /// Reference to the state actor
        state_actor_ref: ActorRef<StateActor>,
        /// Address of the state actor
        state_actor_addr: ChannelAddr,
    },
    /// Send a log message to the state actor
    SendLog {
        /// The target output stream (stdout or stderr)
        output_target: OutputTarget,
        /// The log payload as bytes
        payload: Vec<u8>,
    },
}

/// An actor that manages log writing to state actors
#[derive(Debug)]
#[hyperactor::export(
    handlers = [LogWriterMessage],
)]
pub struct LogWriterActor {
    client: Mailbox,
    /// LogWriterActor will write to the StateActor if the StateActor is set.
    state_actor_ref: Option<ActorRef<StateActor>>,
    mailbox_router: DialMailboxRouter,
    // Sequence counters for stdout and stderr
    stdout_seq: AtomicU64,
    stderr_seq: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct LogWriterActorParams {
    mailbox_router: DialMailboxRouter,
    client: Mailbox,
}

impl LogWriterActorParams {
    // Constructor for LogWriterActorParams
    pub fn new(mailbox_router: DialMailboxRouter, client: Mailbox) -> Self {
        Self {
            mailbox_router,
            client,
        }
    }
}

#[async_trait]
impl Actor for LogWriterActor {
    type Params = LogWriterActorParams;

    async fn new(params: LogWriterActorParams) -> Result<Self, anyhow::Error> {
        Ok(Self {
            client: params.client,
            state_actor_ref: None,
            mailbox_router: params.mailbox_router,
            stdout_seq: AtomicU64::new(0),
            stderr_seq: AtomicU64::new(0),
        })
    }
}

#[async_trait]
#[hyperactor::forward(LogWriterMessage)]
impl LogWriterMessageHandler for LogWriterActor {
    async fn set_state_actor(
        &mut self,
        _ctx: &HyperactorContext<Self>,
        state_actor_ref: ActorRef<StateActor>,
        state_actor_addr: ChannelAddr,
    ) -> Result<(), anyhow::Error> {
        self.state_actor_ref = Some(state_actor_ref.clone());
        // Bind the state actor's proc_id to its address in the router
        let proc_id = state_actor_ref.actor_id().proc_id();
        self.mailbox_router
            .bind(proc_id.clone().into(), state_actor_addr);
        Ok(())
    }

    async fn send_log(
        &mut self,
        _ctx: &HyperactorContext<Self>,
        output_target: OutputTarget,
        payload: Vec<u8>,
    ) -> Result<(), anyhow::Error> {
        // If there's no state actor, do nothing
        if let Some(state_actor_ref) = &self.state_actor_ref {
            // Serialize the payload directly without converting to string
            let serialized_payload = Serialized::serialize_anon(&payload)?;

            // Get the appropriate sequence number
            let seq = match output_target {
                OutputTarget::Stdout => self
                    .stdout_seq
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                OutputTarget::Stderr => self
                    .stderr_seq
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            };

            // Create the log state object
            let metadata = StateMetadata {
                name: Name::from(output_target),
                kind: Kind::Log,
            };
            let spec = LogSpec {};
            let state = LogState::new(seq, serialized_payload);
            let state_object = StateObject::<LogSpec, LogState>::new(metadata, spec, state);

            // Convert to generic state object and send
            match GenericStateObject::try_from(state_object) {
                Ok(generic_state_object) => {
                    let logs = vec![generic_state_object];
                    if let Err(e) =
                        state_actor_ref.send(&self.client, StateMessage::PushLogs { logs })
                    {
                        tracing::error!("Error sending log to state actor: {}", e);
                    }
                }
                Err(e) => {
                    tracing::error!(
                        "Error converting log state object to generic state object: {}",
                        e
                    );
                }
            }
        }
        Ok(())
    }
}

/// Trait for sending logs to a state actor
#[async_trait]
pub trait LogSender: Send + Sync {
    /// Send a log payload to the state actor
    fn send(&self, target: OutputTarget, payload: Vec<u8>) -> anyhow::Result<()>;
}

/// Represents the target output stream (stdout or stderr)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputTarget {
    /// Standard output stream
    Stdout,
    /// Standard error stream
    Stderr,
}

/// Default implementation of LogSender that connects to a real state actor
#[derive(Clone)]
pub struct StateActorLogSender {
    #[allow(dead_code)]
    log_writer_proc: Proc, // Have to keep Proc in alive.
    log_writer_actor_ref: ActorRef<LogWriterActor>,
    client: hyperactor::Mailbox,
}
impl StateActorLogSender {
    /// Create a new StateActorLogSender
    // Helper async function to spawn the log writer actor
    async fn spawn_log_writer_actor(
        proc: &Proc,
        state_actor_addr: &ChannelAddr,
        log_writer_params: LogWriterActorParams,
    ) -> Result<hyperactor::ActorHandle<LogWriterActor>, anyhow::Error> {
        let addr = ChannelAddr::any(state_actor_addr.transport());
        let (_local_addr, _rx) = channel::serve::<LogWriterMessage>(addr.clone()).await?;
        let log_writer_actor_name = "log_writer_actor";
        proc.spawn(log_writer_actor_name, log_writer_params).await
    }

    pub fn new(
        state_actor_addr: ChannelAddr,
        state_actor_ref: ActorRef<StateActor>,
    ) -> Result<Self, anyhow::Error> {
        // Create a LogWriterActor
        let proc_id = id!(log_writer_proc).random_user_proc();
        let router = DialMailboxRouter::new();
        let proc = Proc::new(proc_id, BoxedMailboxSender::new(router.clone()));
        // Create client mailbox
        let client = proc.attach("client")?;
        let log_writer_params = LogWriterActorParams::new(router, client.clone());

        // Spawn the LogWriterActor using our helper function
        let log_writer_handle = futures::executor::block_on(Self::spawn_log_writer_actor(
            &proc,
            &state_actor_addr,
            log_writer_params,
        ))?;

        let log_writer_actor_ref = log_writer_handle.bind();

        // Set up the state actor in the LogWriterActor
        log_writer_actor_ref.send(
            &client,
            LogWriterMessage::SetStateActor {
                state_actor_ref: state_actor_ref.clone(),
                state_actor_addr: state_actor_addr.clone(),
            },
        )?;

        Ok(Self {
            log_writer_proc: proc,
            log_writer_actor_ref,
            client,
        })
    }
}

impl LogSender for StateActorLogSender {
    fn send(&self, target: OutputTarget, payload: Vec<u8>) -> anyhow::Result<()> {
        // Simply forward the payload to the LogWriterActor
        self.log_writer_actor_ref
            .send(
                &self.client,
                LogWriterMessage::SendLog {
                    output_target: target,
                    payload,
                },
            )
            .map_err(|e| anyhow::anyhow!("Failed to send log: {}", e))
    }
}

/// A custom writer that tees to both stdout/stderr and the state actor.
/// It captures output lines and sends them to a state actor at a specified address.
pub struct LogWriter<T: LogSender + Unpin + 'static, S: io::AsyncWrite + Send + Unpin + 'static> {
    output_target: OutputTarget,
    std_writer: S,
    log_sender: T,
}

/// Helper function to create stdout and stderr LogWriter instances
///
/// # Arguments
///
/// * `state_actor_addr` - The address of the state actor to send logs to
///
/// # Returns
///
/// A tuple of boxed writers for stdout and stderr
pub fn create_log_writers(
    state_actor_addr: ChannelAddr,
    state_actor_ref: ActorRef<StateActor>,
) -> Result<
    (
        Box<dyn io::AsyncWrite + Send + Unpin + 'static>,
        Box<dyn io::AsyncWrite + Send + Unpin + 'static>,
    ),
    anyhow::Error,
> {
    // Create a single StateActorLogSender to be shared between stdout and stderr
    let log_sender = StateActorLogSender::new(state_actor_addr, state_actor_ref)?;

    // Create LogWriter instances for stdout and stderr using the shared log sender
    let stdout_writer = LogWriter::with_default_writer(OutputTarget::Stdout, log_sender.clone())?;
    let stderr_writer = LogWriter::with_default_writer(OutputTarget::Stderr, log_sender)?;

    Ok((Box::new(stdout_writer), Box::new(stderr_writer)))
}

impl<T: LogSender + Unpin + 'static, S: io::AsyncWrite + Send + Unpin + 'static> LogWriter<T, S> {
    /// Creates a new LogWriter.
    ///
    /// # Arguments
    ///
    /// * `output_target` - The target output stream (stdout or stderr)
    /// * `std_writer` - The writer to use for stdout/stderr
    /// * `log_sender` - The log sender to use for sending logs to the state actor
    pub fn new(output_target: OutputTarget, std_writer: S, log_sender: T) -> Self {
        Self {
            output_target,
            std_writer,
            log_sender,
        }
    }
}

impl<T: LogSender + Unpin + 'static> LogWriter<T, Box<dyn io::AsyncWrite + Send + Unpin>> {
    /// Creates a new LogWriter with the default stdout/stderr writer.
    ///
    /// # Arguments
    ///
    /// * `output_target` - The target output stream (stdout or stderr)
    /// * `log_sender` - The log sender to use for sending logs to the state actor
    pub fn with_default_writer(
        output_target: OutputTarget,
        log_sender: T,
    ) -> Result<Self, anyhow::Error> {
        let std_writer: Box<dyn io::AsyncWrite + Send + Unpin> = match output_target {
            OutputTarget::Stdout => Box::new(io::stdout()),
            OutputTarget::Stderr => Box::new(io::stderr()),
        };

        Ok(Self {
            output_target,
            std_writer,
            log_sender,
        })
    }
}

impl<T: LogSender + Unpin + 'static, S: io::AsyncWrite + Send + Unpin + 'static> io::AsyncWrite
    for LogWriter<T, S>
{
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        // Get a mutable reference to the std_writer field
        let this = self.get_mut();

        // First, write to stdout/stderr
        match Pin::new(&mut this.std_writer).poll_write(cx, buf) {
            Poll::Ready(Ok(_)) => {
                // Now process for state actor
                // Forward the buffer directly to the log sender without parsing
                let output_target = this.output_target.clone();
                let data_to_send = buf.to_vec();

                // Use the log sender directly without cloning
                // Since LogSender::send takes &self, we don't need to clone it
                if let Err(e) = this.log_sender.send(output_target, data_to_send) {
                    tracing::error!("error sending log to state actor: {}", e);
                }
                // Return success with the full buffer size
                Poll::Ready(Ok(buf.len()))
            }
            other => other, // Propagate any errors or Pending state
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> {
        let this = self.get_mut();
        Pin::new(&mut this.std_writer).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> {
        let this = self.get_mut();
        Pin::new(&mut this.std_writer).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;

    use tokio::io::AsyncWriteExt;
    use tokio::sync::mpsc;

    use super::*;

    // Mock implementation of AsyncWrite that captures written data
    struct MockWriter {
        data: Arc<Mutex<Vec<u8>>>,
    }

    impl MockWriter {
        fn new() -> (Self, Arc<Mutex<Vec<u8>>>) {
            let data = Arc::new(Mutex::new(Vec::new()));
            (Self { data: data.clone() }, data)
        }
    }

    impl io::AsyncWrite for MockWriter {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<Result<usize, io::Error>> {
            let mut data = self.data.lock().unwrap();
            data.extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), io::Error>> {
            Poll::Ready(Ok(()))
        }
    }

    // Mock implementation of LogSender for testing
    struct MockLogSender {
        log_sender: mpsc::UnboundedSender<(OutputTarget, String)>, // (output_target, content)
    }

    impl MockLogSender {
        fn new(log_sender: mpsc::UnboundedSender<(OutputTarget, String)>) -> Self {
            Self { log_sender }
        }
    }

    impl LogSender for MockLogSender {
        fn send(&self, output_target: OutputTarget, payload: Vec<u8>) -> anyhow::Result<()> {
            // For testing purposes, convert to string if it's valid UTF-8
            let line = match std::str::from_utf8(&payload) {
                Ok(s) => s.to_string(),
                Err(_) => String::from_utf8_lossy(&payload).to_string(),
            };

            self.log_sender
                .send((output_target, line))
                .map_err(|e| anyhow::anyhow!("Failed to send log in test: {}", e))
        }
    }

    #[tokio::test]
    async fn test_log_writer_direct_forwarding() {
        // Create a channel to receive logs
        let (log_sender, mut log_receiver) = mpsc::unbounded_channel();

        // Create a mock log sender
        let mock_log_sender = MockLogSender::new(log_sender);

        // Create a mock writer for stdout
        let (mock_writer, _) = MockWriter::new();
        let std_writer: Box<dyn io::AsyncWrite + Send + Unpin> = Box::new(mock_writer);

        // Create a log writer with the mock sender
        let mut writer = LogWriter::new(OutputTarget::Stdout, std_writer, mock_log_sender);

        // Write some data
        writer.write_all(b"Hello, world!").await.unwrap();
        writer.flush().await.unwrap();

        // Check that the log was sent as is
        let (output_target, content) = log_receiver.recv().await.unwrap();
        assert_eq!(output_target, OutputTarget::Stdout);
        assert_eq!(content, "Hello, world!");

        // Write more data
        writer.write_all(b"\nNext line").await.unwrap();
        writer.flush().await.unwrap();

        // Check that the second chunk was sent as is
        let (output_target, content) = log_receiver.recv().await.unwrap();
        assert_eq!(output_target, OutputTarget::Stdout);
        assert_eq!(content, "\nNext line");
    }

    #[tokio::test]
    async fn test_log_writer_stdout_stderr() {
        // Create a channel to receive logs
        let (log_sender, mut log_receiver) = mpsc::unbounded_channel();

        // Create mock log senders for stdout and stderr
        let stdout_sender = MockLogSender::new(log_sender.clone());
        let stderr_sender = MockLogSender::new(log_sender);

        // Create mock writers for stdout and stderr
        let (stdout_mock_writer, _) = MockWriter::new();
        let stdout_writer: Box<dyn io::AsyncWrite + Send + Unpin> = Box::new(stdout_mock_writer);

        let (stderr_mock_writer, _) = MockWriter::new();
        let stderr_writer: Box<dyn io::AsyncWrite + Send + Unpin> = Box::new(stderr_mock_writer);

        // Create log writers with the mock senders
        let mut stdout_writer = LogWriter::new(OutputTarget::Stdout, stdout_writer, stdout_sender);
        let mut stderr_writer = LogWriter::new(OutputTarget::Stderr, stderr_writer, stderr_sender);

        // Write to stdout and stderr
        stdout_writer.write_all(b"Stdout data").await.unwrap();
        stdout_writer.flush().await.unwrap();

        stderr_writer.write_all(b"Stderr data").await.unwrap();
        stderr_writer.flush().await.unwrap();

        // Check that logs were sent with correct output targets
        // Note: We can't guarantee the order of reception since they're sent from different tasks
        let mut received_stdout = false;
        let mut received_stderr = false;

        for _ in 0..2 {
            let (output_target, content) = log_receiver.recv().await.unwrap();
            match output_target {
                OutputTarget::Stdout => {
                    assert_eq!(content, "Stdout data");
                    received_stdout = true;
                }
                OutputTarget::Stderr => {
                    assert_eq!(content, "Stderr data");
                    received_stderr = true;
                }
            }
        }

        assert!(received_stdout);
        assert!(received_stderr);
    }

    #[tokio::test]
    async fn test_log_writer_binary_data() {
        // Create a channel to receive logs
        let (log_sender, mut log_receiver) = mpsc::unbounded_channel();

        // Create a mock log sender
        let mock_log_sender = MockLogSender::new(log_sender);

        // Create a mock writer for stdout
        let (mock_writer, _) = MockWriter::new();
        let std_writer: Box<dyn io::AsyncWrite + Send + Unpin> = Box::new(mock_writer);

        // Create a log writer with the mock sender
        let mut writer = LogWriter::new(OutputTarget::Stdout, std_writer, mock_log_sender);

        // Write binary data (including non-UTF8 bytes)
        let binary_data = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F, 0xFF, 0xFE, 0x00];
        writer.write_all(&binary_data).await.unwrap();
        writer.flush().await.unwrap();

        // Check that the log was sent and converted to string (with lossy UTF-8 conversion in MockLogSender)
        let (output_target, content) = log_receiver.recv().await.unwrap();
        assert_eq!(output_target, OutputTarget::Stdout);
        // The content should be "Hello" followed by replacement characters for invalid bytes
        assert!(content.starts_with("Hello"));
        // The rest of the content will be replacement characters, but we don't care about the exact representation
    }
}
