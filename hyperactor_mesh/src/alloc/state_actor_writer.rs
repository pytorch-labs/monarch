/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use async_trait::async_trait;
use hyperactor::ActorRef;
use hyperactor::channel::ChannelAddr;
use hyperactor::id;
use hyperactor_state::create_remote_client;
use hyperactor_state::object::GenericStateObject;
use hyperactor_state::object::LogSpec;
use hyperactor_state::object::LogState;
use hyperactor_state::object::StateMetadata;
use hyperactor_state::object::StateObject;
use hyperactor_state::state_actor::StateActor;
use hyperactor_state::state_actor::StateMessage;
use tokio::io;

/// Trait for sending logs to a state actor
#[async_trait]
pub trait StateActorIngestor: Send + Sync {
    /// Send a log line to the state actor
    async fn send_log(&self, is_stdout: bool, line: String) -> Result<(), String>;
}

/// Represents the target output stream (stdout or stderr)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTarget {
    /// Standard output stream
    Stdout,
    /// Standard error stream
    Stderr,
}

/// Default implementation of StateActorIngestor that connects to a real state actor
pub struct StateActorIngestorImpl {
    state_actor_addr: ChannelAddr,
    remote_client: tokio::sync::Mutex<Option<(hyperactor::proc::Proc, hyperactor::Mailbox)>>,
    state_actor_ref: ActorRef<StateActor>,
}

impl StateActorIngestorImpl {
    /// Create a new StateActorIngestorImpl
    pub fn new(state_actor_addr: ChannelAddr) -> Self {
        let state_actor_ref = ActorRef::<StateActor>::attest(id!(state_server[0].state[0]));
        Self {
            state_actor_addr,
            remote_client: tokio::sync::Mutex::new(None),
            state_actor_ref,
        }
    }
}

#[async_trait]
impl StateActorIngestor for StateActorIngestorImpl {
    async fn send_log(&self, is_stdout: bool, line: String) -> Result<(), String> {
        // Get or create the remote client
        let mut client_guard = self.remote_client.lock().await;
        if client_guard.is_none() {
            *client_guard = Some(
                create_remote_client(self.state_actor_addr.clone())
                    .await
                    .map_err(|e| format!("Failed to connect to state actor: {}", e))?,
            );
        }

        let (_, remote_client) = client_guard.as_ref().unwrap();

        // Set up sequence counters
        static STDOUT_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static STDERR_SEQ: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(10000);

        // Get the appropriate sequence number
        let seq = if is_stdout {
            STDOUT_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        } else {
            STDERR_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        };

        // Create the log state object
        let metadata = StateMetadata {
            name: if is_stdout { "stdout" } else { "stderr" }.to_string(),
            kind: "log".to_string(),
        };
        let spec = LogSpec {};
        let state = LogState::new(seq, line);
        let state_object = StateObject::<LogSpec, LogState>::new(metadata, spec, state);

        // Convert to generic state object and send
        let generic_state_object = GenericStateObject::try_from(state_object)
            .map_err(|e| format!("Failed to convert state object: {}", e))?;

        let logs = vec![generic_state_object];
        self.state_actor_ref
            .send(remote_client, StateMessage::PushLogs { logs })
            .map_err(|e| format!("Error sending log to state actor: {}", e))?;

        Ok(())
    }
}

/// A custom writer that tees to both stdout/stderr and the state actor.
/// It captures output lines and sends them to a state actor at a specified address.
pub struct StateActorWriter {
    output_target: OutputTarget,
    buffer: String,
    std_writer: Box<dyn io::AsyncWrite + Send + Unpin>,
    state_actor_ingestor: Arc<dyn StateActorIngestor>,
}

impl StateActorWriter {
    /// Creates a new StateActorWriter with the default state actor ingestor.
    ///
    /// # Arguments
    ///
    /// * `output_target` - The target output stream (stdout or stderr)
    /// * `state_actor_addr` - The address of the state actor to send logs to
    pub fn new(output_target: OutputTarget, state_actor_addr: ChannelAddr) -> Self {
        let std_writer: Box<dyn io::AsyncWrite + Send + Unpin> = match output_target {
            OutputTarget::Stdout => Box::new(io::stdout()),
            OutputTarget::Stderr => Box::new(io::stderr()),
        };

        let state_actor_ingestor = Arc::new(StateActorIngestorImpl::new(state_actor_addr));

        Self {
            output_target,
            buffer: String::new(),
            std_writer,
            state_actor_ingestor,
        }
    }

    /// Creates a new StateActorWriter with a custom state actor ingestor.
    ///
    /// # Arguments
    ///
    /// * `output_target` - The target output stream (stdout or stderr)
    /// * `std_writer` - The writer to use for stdout/stderr
    /// * `state_actor_ingestor` - The ingestor to use for sending logs to the state actor
    #[allow(dead_code)]
    pub fn with_client(
        output_target: OutputTarget,
        std_writer: Box<dyn io::AsyncWrite + Send + Unpin>,
        state_actor_ingestor: Arc<dyn StateActorIngestor>,
    ) -> Self {
        Self {
            output_target,
            buffer: String::new(),
            std_writer,
            state_actor_ingestor,
        }
    }
}

impl io::AsyncWrite for StateActorWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        // First, write to stdout/stderr
        match Pin::new(&mut self.std_writer).poll_write(cx, buf) {
            Poll::Ready(Ok(_)) => {
                // Now process for state actor
                if let Ok(s) = std::str::from_utf8(buf) {
                    self.buffer.push_str(s);

                    // If we have a complete line, process it
                    if self.buffer.contains('\n') {
                        let lines: Vec<String> = self
                            .buffer
                            .split('\n')
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .collect();

                        if !lines.is_empty() {
                            // Clone the data we need for the async task
                            let is_stdout = matches!(self.output_target, OutputTarget::Stdout);
                            let lines_to_send = lines.clone();

                            // Clone the state actor ingestor for the async task
                            let state_actor_ingestor = self.state_actor_ingestor.clone();

                            // Spawn a task to send the logs to the state actor
                            tokio::spawn(async move {
                                for line in lines_to_send {
                                    if let Err(e) =
                                        state_actor_ingestor.send_log(is_stdout, line).await
                                    {
                                        eprintln!("Error sending log to state actor: {}", e);
                                    }
                                }
                            });
                        }

                        // Keep any remaining partial line
                        if self.buffer.ends_with('\n') {
                            self.buffer.clear();
                        } else {
                            let last_newline = self.buffer.rfind('\n').map_or(0, |i| i + 1);
                            self.buffer = self.buffer[last_newline..].to_string();
                        }
                    }
                }

                // Return success with the full buffer size
                Poll::Ready(Ok(buf.len()))
            }
            other => other, // Propagate any errors or Pending state
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> {
        Pin::new(&mut self.std_writer).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        Pin::new(&mut self.std_writer).poll_shutdown(cx)
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

    // Mock implementation of StateActorIngestor for testing
    struct MockStateActorIngestor {
        log_sender: mpsc::UnboundedSender<(bool, String)>, // (is_stdout, content)
    }

    impl MockStateActorIngestor {
        fn new(log_sender: mpsc::UnboundedSender<(bool, String)>) -> Self {
            Self { log_sender }
        }
    }

    #[async_trait]
    impl StateActorIngestor for MockStateActorIngestor {
        async fn send_log(&self, is_stdout: bool, line: String) -> Result<(), String> {
            self.log_sender
                .send((is_stdout, line))
                .map_err(|e| e.to_string())
        }
    }

    #[tokio::test]
    async fn test_state_actor_writer_complete_lines() {
        // Create a channel to receive logs
        let (log_sender, mut log_receiver) = mpsc::unbounded_channel();

        // Create a mock state actor ingestor
        let state_actor_ingestor = Arc::new(MockStateActorIngestor::new(log_sender));

        // Create a mock writer for stdout
        let (mock_writer, _) = MockWriter::new();
        let std_writer: Box<dyn io::AsyncWrite + Send + Unpin> = Box::new(mock_writer);

        // Create a state actor writer with the mock ingestor
        let mut writer =
            StateActorWriter::with_client(OutputTarget::Stdout, std_writer, state_actor_ingestor);

        // Write a complete line
        writer.write_all(b"Hello, world!\n").await.unwrap();
        writer.flush().await.unwrap();

        // Check that the log was sent
        let (is_stdout, content) = log_receiver.recv().await.unwrap();
        assert!(is_stdout);
        assert_eq!(content, "Hello, world!");
    }

    #[tokio::test]
    async fn test_state_actor_writer_partial_lines() {
        // Create a channel to receive logs
        let (log_sender, mut log_receiver) = mpsc::unbounded_channel();

        // Create a mock state actor ingestor
        let state_actor_ingestor = Arc::new(MockStateActorIngestor::new(log_sender));

        // Create a mock writer for stdout
        let (mock_writer, _) = MockWriter::new();
        let std_writer: Box<dyn io::AsyncWrite + Send + Unpin> = Box::new(mock_writer);

        // Create a state actor writer with the mock ingestor
        let mut writer =
            StateActorWriter::with_client(OutputTarget::Stdout, std_writer, state_actor_ingestor);

        // Write a partial line
        writer.write_all(b"Hello, ").await.unwrap();
        writer.flush().await.unwrap();

        // No log should be sent yet
        assert!(log_receiver.try_recv().is_err());

        // Complete the line
        writer.write_all(b"world!\n").await.unwrap();
        writer.flush().await.unwrap();

        // Now the log should be sent
        let (is_stdout, content) = log_receiver.recv().await.unwrap();
        assert!(is_stdout);
        assert_eq!(content, "Hello, world!");
    }

    #[tokio::test]
    async fn test_state_actor_writer_multiple_lines() {
        // Create a channel to receive logs
        let (log_sender, mut log_receiver) = mpsc::unbounded_channel();

        // Create a mock state actor ingestor
        let state_actor_ingestor = Arc::new(MockStateActorIngestor::new(log_sender));

        // Create a mock writer for stdout
        let (mock_writer, _) = MockWriter::new();
        let std_writer: Box<dyn io::AsyncWrite + Send + Unpin> = Box::new(mock_writer);

        // Create a state actor writer with the mock ingestor
        let mut writer =
            StateActorWriter::with_client(OutputTarget::Stdout, std_writer, state_actor_ingestor);

        // Write multiple lines
        writer.write_all(b"Line 1\nLine 2\nLine 3\n").await.unwrap();
        writer.flush().await.unwrap();

        // Check that all logs were sent
        let (is_stdout, content) = log_receiver.recv().await.unwrap();
        assert!(is_stdout);
        assert_eq!(content, "Line 1");

        let (is_stdout, content) = log_receiver.recv().await.unwrap();
        assert!(is_stdout);
        assert_eq!(content, "Line 2");

        let (is_stdout, content) = log_receiver.recv().await.unwrap();
        assert!(is_stdout);
        assert_eq!(content, "Line 3");
    }

    #[tokio::test]
    async fn test_state_actor_writer_stdout_stderr() {
        // Create a channel to receive logs
        let (log_sender, mut log_receiver) = mpsc::unbounded_channel();

        // Create mock state actor ingestors for stdout and stderr
        let stdout_ingestor = Arc::new(MockStateActorIngestor::new(log_sender.clone()));
        let stderr_ingestor = Arc::new(MockStateActorIngestor::new(log_sender));

        // Create mock writers for stdout and stderr
        let (stdout_mock_writer, _) = MockWriter::new();
        let stdout_writer: Box<dyn io::AsyncWrite + Send + Unpin> = Box::new(stdout_mock_writer);

        let (stderr_mock_writer, _) = MockWriter::new();
        let stderr_writer: Box<dyn io::AsyncWrite + Send + Unpin> = Box::new(stderr_mock_writer);

        // Create state actor writers with the mock ingestors
        let mut stdout_writer =
            StateActorWriter::with_client(OutputTarget::Stdout, stdout_writer, stdout_ingestor);
        let mut stderr_writer =
            StateActorWriter::with_client(OutputTarget::Stderr, stderr_writer, stderr_ingestor);

        // Write to stdout and stderr
        stdout_writer.write_all(b"Stdout line\n").await.unwrap();
        stdout_writer.flush().await.unwrap();

        stderr_writer.write_all(b"Stderr line\n").await.unwrap();
        stderr_writer.flush().await.unwrap();

        // Check that logs were sent with correct is_stdout flags
        // Note: We can't guarantee the order of reception since they're sent from different tasks
        let mut received_stdout = false;
        let mut received_stderr = false;

        for _ in 0..2 {
            let (is_stdout, content) = log_receiver.recv().await.unwrap();
            if is_stdout {
                assert_eq!(content, "Stdout line");
                received_stdout = true;
            } else {
                assert_eq!(content, "Stderr line");
                received_stderr = true;
            }
        }

        assert!(received_stdout);
        assert!(received_stderr);
    }

    #[tokio::test]
    async fn test_state_actor_writer_partial_line_at_end() {
        // Create a channel to receive logs
        let (log_sender, mut log_receiver) = mpsc::unbounded_channel();

        // Create a mock state actor ingestor
        let state_actor_ingestor = Arc::new(MockStateActorIngestor::new(log_sender));

        // Create a mock writer for stdout
        let (mock_writer, _) = MockWriter::new();
        let std_writer: Box<dyn io::AsyncWrite + Send + Unpin> = Box::new(mock_writer);

        // Create a state actor writer with the mock ingestor
        let mut writer =
            StateActorWriter::with_client(OutputTarget::Stdout, std_writer, state_actor_ingestor);

        // Write a complete line followed by a partial line
        writer.write_all(b"Line 1\nPartial").await.unwrap();
        writer.flush().await.unwrap();

        // Only the complete line should be sent
        let (is_stdout, content) = log_receiver.recv().await.unwrap();
        assert!(is_stdout);
        assert_eq!(content, "Line 1");

        // No more logs should be available
        assert!(log_receiver.try_recv().is_err());

        // Complete the partial line
        writer.write_all(b" line\n").await.unwrap();
        writer.flush().await.unwrap();

        // Now the second log should be sent
        let (is_stdout, content) = log_receiver.recv().await.unwrap();
        assert!(is_stdout);
        assert_eq!(content, "Partial line");
    }
}
