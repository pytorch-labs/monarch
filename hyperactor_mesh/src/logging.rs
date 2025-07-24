/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context as TaskContext;
use std::task::Poll;
use std::time::Duration;
use std::time::SystemTime;

use anyhow::Error;
use anyhow::Result;
use async_trait::async_trait;
use chrono::DateTime;
use chrono::Local;
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
use hyperactor::channel::ChannelTx;
use hyperactor::channel::Rx;
use hyperactor::channel::Tx;
use hyperactor::channel::TxStatus;
use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;
use hyperactor::data::Serialized;
use hyperactor::message::Bind;
use hyperactor::message::Bindings;
use hyperactor::message::Unbind;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use tokio::io;
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio::sync::watch::Receiver;
use tokio::task::JoinHandle;

use crate::bootstrap::BOOTSTRAP_LOG_CHANNEL;

const DEFAULT_AGGREGATE_WINDOW_SEC: u64 = 5;

#[derive(Debug, Clone, PartialEq)]
/// Token represents a single token in a log line. It can be either a string or a number.
enum Token {
    String(String),
    Number((f64, f64)), // (min, max)
}

impl Token {
    // Helper function to create a Number token with the same min and max value
    pub fn number(value: f64) -> Self {
        Token::Number((value, value))
    }
}

#[derive(Debug, Clone)]
/// LogLine represents a single log line. It contains a list of tokens and a count of how many times the same
/// log line has been recorded. Here "same" is defined in PartialEq.
struct LogLine {
    tokens: Vec<Token>,
    pub count: u64,
}

impl LogLine {
    pub fn try_merge(&mut self, other: LogLine) -> anyhow::Result<()> {
        // Check if they have the same length
        if self.tokens.len() != other.tokens.len() {
            return Err(anyhow::anyhow!(
                "cannot merge LogLines with different lengths"
            ));
        }
        // Check each token and merge if possible
        for (i, (self_token, other_token)) in
            self.tokens.iter_mut().zip(other.tokens.iter()).enumerate()
        {
            match (self_token, other_token) {
                // For String tokens, the strings must match
                (Token::String(self_string), Token::String(other_string)) => {
                    if self_string != other_string {
                        return Err(anyhow::anyhow!(
                            "cannot merge LogLines with different String tokens at position {}: {}, {}",
                            i,
                            self_string,
                            other_string
                        ));
                    }
                }
                // For Number tokens, update min and max
                (Token::Number(self_num), Token::Number(other_num)) => {
                    // Update min (take the smaller of the two mins)
                    self_num.0 = self_num.0.min(other_num.0);
                    // Update max (take the larger of the two maxes)
                    self_num.1 = self_num.1.max(other_num.1);
                }
                // If one is String and the other is Number, they're not mergeable
                _ => {
                    return Err(anyhow::anyhow!(
                        "cannot merge LogLines with different token types at position {}",
                        i
                    ));
                }
            }
        }

        // Increment the count when merging
        self.count += other.count;

        Ok(())
    }
}

fn parse_line(line: &str) -> LogLine {
    let mut result_tokens = Vec::new();

    // Regex to match number followed by optional string: captures number and remaining string separately
    let number_string_regex = Regex::new(r"^(-?\d+(?:\.\d+)?)(.*)$").unwrap();

    // Split by whitespace first
    for token in line.split_whitespace() {
        if let Some(captures) = number_string_regex.captures(token) {
            let number_part = captures.get(1).unwrap().as_str();
            let string_part = captures.get(2).unwrap().as_str();

            // Parse the number part
            if let Ok(n) = number_part.parse::<f64>() {
                result_tokens.push(Token::number(n));

                // Add string part if it's not empty
                if !string_part.is_empty() {
                    result_tokens.push(Token::String(string_part.to_string()));
                }
            } else {
                // Fallback: treat entire token as string
                result_tokens.push(Token::String(token.to_string()));
            }
        } else {
            // No number at start, treat as string
            result_tokens.push(Token::String(token.to_string()));
        }
    }

    LogLine {
        tokens: result_tokens,
        count: 1, // Initialize count to 1 for the first conversion
    }
}

#[derive(Debug, Clone)]
/// Aggregator is a struct that holds a list of LogLines and a start time. It can aggregate new log lines to
/// existing ones if they are "similar" (same strings, different numbers).
struct Aggregator {
    lines: Vec<LogLine>,
    start_time: SystemTime,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::String(s) => write!(f, "{}", s),
            Token::Number((min, max)) => write!(f, "<<{}, {}>>", min, max),
        }
    }
}

impl fmt::Display for LogLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tokens_str: Vec<String> = self.tokens.iter().map(|t| t.to_string()).collect();
        write!(
            f,
            "\x1b[33m[{} processes]\x1b[0m {}",
            self.count,
            tokens_str.join(" ")
        )
    }
}

impl Aggregator {
    pub fn new() -> Self {
        Aggregator {
            lines: vec![],
            start_time: RealClock.system_time_now(),
        }
    }

    pub fn reset(&mut self) {
        self.lines.clear();
        self.start_time = RealClock.system_time_now();
    }

    pub fn add_line(&mut self, line: &str) -> anyhow::Result<()> {
        // 1. Convert the string into a LogLine
        let new_line = parse_line(line);

        // 2. Iterate through existing lines and try to merge
        for existing_line in &mut self.lines {
            // Try to merge directly without checking equality first
            if existing_line.try_merge(new_line.clone()).is_ok() {
                return Ok(());
            }
        }

        // 3. If no merge succeeds, append the new line
        self.lines.push(new_line);
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

// Helper function to format SystemTime
fn format_system_time(time: SystemTime) -> String {
    let datetime: DateTime<Local> = time.into();
    datetime.format("%Y-%m-%d %H:%M:%S").to_string()
}

impl fmt::Display for Aggregator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format the start time
        let start_time_str = format_system_time(self.start_time);

        // Get and format the current time
        let current_time = RealClock.system_time_now();
        let end_time_str = format_system_time(current_time);

        // Write the header with formatted time window
        writeln!(
            f,
            "\x1b[36m>>> Aggregated Logs ({}) >>>\x1b[0m",
            start_time_str
        )?;

        // Write each log line
        for line in self.lines.iter() {
            writeln!(f, "{}", line)?;
        }
        writeln!(
            f,
            "\x1b[36m<<< Aggregated Logs ({}) <<<\x1b[0m",
            end_time_str
        )?;
        Ok(())
    }
}

/// Messages that can be sent to the LogClientActor remotely.
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
pub enum LogMessage {
    /// Log details
    Log {
        /// The hostname of the process that generated the log
        hostname: String,
        /// The pid of the process that generated the log
        pid: u32,
        /// The target output stream (stdout or stderr)
        output_target: OutputTarget,
        /// The log payload as bytes
        payload: Serialized,
    },
}

/// Messages that can be sent to the LogClient locally.
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
pub enum LogClientMessage {
    SetAggregate {
        /// The time window in seconds to aggregate logs. If None, aggregation is disabled.
        aggregate_window_sec: Option<u64>,
    },
}

/// Trait for sending logs
#[async_trait]
pub trait LogSender: Send + Sync {
    /// Send a log payload in bytes
    fn send(&mut self, target: OutputTarget, payload: Vec<u8>) -> anyhow::Result<()>;
}

/// Represents the target output stream (stdout or stderr)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum OutputTarget {
    /// Standard output stream
    Stdout,
    /// Standard error stream
    Stderr,
}

/// Write the log to a local unix channel so some actors can listen to it and stream the log back.
#[derive(Clone)]
pub struct LocalLogSender {
    hostname: String,
    pid: u32,
    tx: Arc<ChannelTx<LogMessage>>,
    status: Receiver<TxStatus>,
}

impl LocalLogSender {
    fn new(log_channel: ChannelAddr, pid: u32) -> Result<Self, anyhow::Error> {
        let tx = channel::dial::<LogMessage>(log_channel)?;
        let status = tx.status().clone();

        let hostname = hostname::get()
            .unwrap_or_else(|_| "unknown_host".into())
            .into_string()
            .unwrap_or("unknown_host".to_string());
        Ok(Self {
            hostname,
            pid,
            tx: Arc::new(tx),
            status,
        })
    }
}

impl LogSender for LocalLogSender {
    fn send(&mut self, target: OutputTarget, payload: Vec<u8>) -> anyhow::Result<()> {
        if TxStatus::Active == *self.status.borrow() {
            self.tx.post(LogMessage::Log {
                hostname: self.hostname.clone(),
                pid: self.pid,
                output_target: target,
                payload: Serialized::serialize_anon(&payload)?,
            });
        } else {
            tracing::trace!(
                "log sender {} is not active, skip sending log",
                self.tx.addr()
            )
        }

        Ok(())
    }
}

/// A custom writer that tees to both stdout/stderr.
/// It captures output lines and sends them to the child process.
pub struct LogWriter<T: LogSender + Unpin + 'static, S: io::AsyncWrite + Send + Unpin + 'static> {
    output_target: OutputTarget,
    std_writer: S,
    log_sender: T,
}

/// Helper function to create stdout and stderr LogWriter instances
///
/// # Arguments
///
/// * `log_channel` - The unix channel for the writer to stream logs to
/// * `pid` - The process ID of the process
///
/// # Returns
///
/// A tuple of boxed writers for stdout and stderr
pub fn create_log_writers(
    log_channel: ChannelAddr,
    pid: u32,
) -> Result<
    (
        Box<dyn io::AsyncWrite + Send + Unpin + 'static>,
        Box<dyn io::AsyncWrite + Send + Unpin + 'static>,
    ),
    anyhow::Error,
> {
    let log_sender = LocalLogSender::new(log_channel, pid)?;

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
    /// * `log_sender` - The log sender to use for sending logs
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
    /// * `log_sender` - The log sender to use for sending logs
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
        cx: &mut TaskContext<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        // Get a mutable reference to the std_writer field
        let this = self.get_mut();

        // First, write to stdout/stderr
        match Pin::new(&mut this.std_writer).poll_write(cx, buf) {
            Poll::Ready(Ok(_)) => {
                // Forward the buffer directly to the log sender without parsing
                let output_target = this.output_target.clone();
                let data_to_send = buf.to_vec();

                // Use the log sender directly without cloning
                // Since LogSender::send takes &self, we don't need to clone it
                if let Err(e) = this.log_sender.send(output_target, data_to_send) {
                    tracing::error!("error sending log: {}", e);
                }
                // Return success with the full buffer size
                Poll::Ready(Ok(buf.len()))
            }
            other => other, // Propagate any errors or Pending state
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Result<(), io::Error>> {
        let this = self.get_mut();
        Pin::new(&mut this.std_writer).poll_flush(cx)
    }

    fn poll_shutdown(
        self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<Result<(), io::Error>> {
        let this = self.get_mut();
        Pin::new(&mut this.std_writer).poll_shutdown(cx)
    }
}

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

/// A client to receive logs from remote processes
#[derive(Debug)]
#[hyperactor::export(
    spawn = true,
    handlers = [LogMessage, LogClientMessage],
)]
pub struct LogClientActor {
    log_tx: mpsc::Sender<(OutputTarget, String)>,
    #[allow(unused)]
    aggregator_handle: JoinHandle<Result<(), Error>>,
    /// The watch sender for the aggregation window in seconds
    aggregate_window_tx: watch::Sender<u64>,
    should_aggregate: bool,
}

#[async_trait]
impl Actor for LogClientActor {
    /// The aggregation window in seconds.
    type Params = ();

    async fn new(_: ()) -> Result<Self, anyhow::Error> {
        // Create mpsc channel for log messages
        let (log_tx, log_rx) = mpsc::channel::<(OutputTarget, String)>(1000);

        // Create a watch channel for the aggregation window
        let (aggregate_window_tx, aggregate_window_rx) =
            watch::channel(DEFAULT_AGGREGATE_WINDOW_SEC);

        // Start the loggregator
        let aggregator_handle =
            { tokio::spawn(async move { start_aggregator(log_rx, aggregate_window_rx).await }) };

        Ok(Self {
            log_tx,
            aggregator_handle,
            aggregate_window_tx,
            should_aggregate: false,
        })
    }
}

async fn start_aggregator(
    mut log_rx: mpsc::Receiver<(OutputTarget, String)>,
    mut interval_sec_rx: watch::Receiver<u64>,
) -> anyhow::Result<()> {
    let mut aggregators = HashMap::new();
    aggregators.insert(OutputTarget::Stderr, Aggregator::new());
    aggregators.insert(OutputTarget::Stdout, Aggregator::new());
    let mut interval =
        tokio::time::interval(tokio::time::Duration::from_secs(*interval_sec_rx.borrow()));

    // Start the event loop
    loop {
        tokio::select! {
            // Process incoming log messages
            Some((output_target, log_line)) = log_rx.recv() => {
                if let Some(aggregator) = aggregators.get_mut(&output_target) {
                    if let Err(e) = aggregator.add_line(&log_line) {
                        tracing::error!("error adding log line: {}", e);
                    }
                } else {
                    tracing::error!("unknown output target: {:?}", output_target);
                }
            }
            // Watch for changes in the interval
            Ok(_) = interval_sec_rx.changed() => {
                interval = tokio::time::interval(tokio::time::Duration::from_secs(*interval_sec_rx.borrow()));
            }

            // Every interval tick, print and reset the aggregator
            _ = interval.tick() => {
                for (output_target, aggregator) in aggregators.iter_mut() {
                    if aggregator.is_empty() {
                        continue;
                    }
                    if output_target == &OutputTarget::Stdout {
                        println!("{}", aggregator);
                    } else {
                        eprintln!("{}", aggregator);
                    }

                    // Reset the aggregator
                    aggregator.reset();
                }
            }

            // Exit if the channel is closed
            else => {
                tracing::error!("log channel closed, exiting aggregator");
                break;
            }
        }
    }

    Ok(())
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

        for line in message_lines {
            if self.should_aggregate {
                self.log_tx.send((output_target, line)).await?;
            } else {
                let message = format!("[{} {}] {}", hostname, pid, line);
                match output_target {
                    OutputTarget::Stdout => println!("{}", message),
                    OutputTarget::Stderr => eprintln!("{}", message),
                    _ => {
                        tracing::error!("unknown output target: {:?}", output_target);
                        println!("{}", message);
                    }
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
#[hyperactor::forward(LogClientMessage)]
impl LogClientMessageHandler for LogClientActor {
    async fn set_aggregate(
        &mut self,
        _cx: &Context<Self>,
        aggregate_window_sec: Option<u64>,
    ) -> Result<(), anyhow::Error> {
        if let Some(window) = aggregate_window_sec {
            // Send the new value through the watch channel
            self.aggregate_window_tx.send(window)?;
        }
        self.should_aggregate = aggregate_window_sec.is_some();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;

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
    use tokio::io::AsyncWriteExt;
    use tokio::sync::mpsc;

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
            _cx: &mut TaskContext<'_>,
            buf: &[u8],
        ) -> Poll<Result<usize, io::Error>> {
            let mut data = self.data.lock().unwrap();
            data.extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _cx: &mut TaskContext<'_>,
        ) -> Poll<Result<(), io::Error>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(
            self: Pin<&mut Self>,
            _cx: &mut TaskContext<'_>,
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
        fn send(&mut self, output_target: OutputTarget, payload: Vec<u8>) -> anyhow::Result<()> {
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

    #[test]
    fn test_try_merge_successful_with_same_strings_and_numbers() {
        let mut line1 = LogLine {
            tokens: vec![
                Token::String("ERROR".to_string()),
                Token::Number((10.0, 15.0)),
                Token::String("timeout".to_string()),
            ],
            count: 1,
        };
        let line2 = LogLine {
            tokens: vec![
                Token::String("ERROR".to_string()),
                Token::Number((12.0, 20.0)),
                Token::String("timeout".to_string()),
            ],
            count: 1,
        };

        let result = line1.try_merge(line2);
        assert!(result.is_ok());

        // Check that the number range was updated correctly
        if let Token::Number((min, max)) = &line1.tokens[1] {
            assert_eq!(*min, 10.0); // min of 10.0 and 12.0
            assert_eq!(*max, 20.0); // max of 15.0 and 20.0
        } else {
            panic!("expected number token");
        }

        // Check that the count was incremented
        assert_eq!(line1.count, 2);
    }

    #[test]
    fn test_try_merge_fails_with_different_lengths() {
        let mut line1 = LogLine {
            tokens: vec![Token::String("ERROR".to_string())],
            count: 1,
        };
        let line2 = LogLine {
            tokens: vec![
                Token::String("ERROR".to_string()),
                Token::String("timeout".to_string()),
            ],
            count: 1,
        };

        let result = line1.try_merge(line2);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("different lengths")
        );
    }

    #[test]
    fn test_try_merge_fails_with_different_strings() {
        let mut line1 = LogLine {
            tokens: vec![
                Token::String("ERROR".to_string()),
                Token::String("timeout".to_string()),
            ],
            count: 1,
        };
        let line2 = LogLine {
            tokens: vec![
                Token::String("ERROR".to_string()),
                Token::String("connection".to_string()),
            ],
            count: 1,
        };

        let result = line1.try_merge(line2);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("different String tokens")
        );
    }

    #[test]
    fn test_try_merge_fails_with_mixed_token_types() {
        let mut line1 = LogLine {
            tokens: vec![
                Token::String("ERROR".to_string()),
                Token::Number((10.0, 10.0)),
            ],
            count: 1,
        };
        let line2 = LogLine {
            tokens: vec![
                Token::String("ERROR".to_string()),
                Token::String("timeout".to_string()),
            ],
            count: 1,
        };

        let result = line1.try_merge(line2);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("different token types")
        );
    }

    #[test]
    fn test_add_line_to_empty_aggregator() {
        let mut aggregator = Aggregator::new();
        let result = aggregator.add_line("ERROR 404 not found");

        assert!(result.is_ok());
        assert_eq!(aggregator.lines.len(), 1);
        assert_eq!(aggregator.lines[0].tokens.len(), 4);
    }

    #[test]
    fn test_add_line_merges_with_existing_line() {
        let mut aggregator = Aggregator::new();

        // Add first line
        aggregator.add_line("ERROR 404 timeout").unwrap();
        assert_eq!(aggregator.lines.len(), 1);

        // Add second line that should merge (same strings, different number)
        aggregator.add_line("ERROR 500 timeout").unwrap();
        assert_eq!(aggregator.lines.len(), 1); // Should still be 1 line after merge

        // Check that the number range was updated
        if let Token::Number((min, max)) = &aggregator.lines[0].tokens[1] {
            assert_eq!(*min, 404.0);
            assert_eq!(*max, 500.0);
        } else {
            panic!("expected number token");
        }

        // Check that the count was incremented
        assert_eq!(aggregator.lines[0].count, 2);
    }

    #[test]
    fn test_add_line_creates_separate_line_when_no_merge_possible() {
        let mut aggregator = Aggregator::new();

        // Add first line
        aggregator.add_line("ERROR 404 timeout").unwrap();
        assert_eq!(aggregator.lines.len(), 1);

        // Add second line that cannot merge (different string)
        aggregator.add_line("ERROR 404 connection").unwrap();
        assert_eq!(aggregator.lines.len(), 2); // Should be 2 lines now

        // Add third line that merges with first
        aggregator.add_line("ERROR 500 timeout").unwrap();
        assert_eq!(aggregator.lines.len(), 2); // Should still be 2 lines

        // Check that first line's number range was updated
        if let Token::Number((min, max)) = &aggregator.lines[0].tokens[1] {
            assert_eq!(*min, 404.0);
            assert_eq!(*max, 500.0);
        } else {
            panic!("expected number token");
        }

        // Check that the count was incremented for the first line
        assert_eq!(aggregator.lines[0].count, 2);
        // Check that the second line still has count 1
        assert_eq!(aggregator.lines[1].count, 1);
    }

    #[test]
    fn test_parse_line_with_mixed_tokens() {
        let line = parse_line("ERROR 404 not found");

        assert_eq!(line.tokens.len(), 4);

        // Check each token
        match &line.tokens[0] {
            Token::String(s) => assert_eq!(s, "ERROR"),
            _ => panic!("expected string token"),
        }

        match &line.tokens[1] {
            Token::Number((min, max)) => {
                assert_eq!(*min, 404.0);
                assert_eq!(*max, 404.0);
            }
            _ => panic!("expected number token"),
        }

        match &line.tokens[2] {
            Token::String(s) => assert_eq!(s, "not"),
            _ => panic!("expected string token"),
        }

        match &line.tokens[3] {
            Token::String(s) => assert_eq!(s, "found"),
            _ => panic!("expected string token"),
        }

        // Check that count is initialized to 1
        assert_eq!(line.count, 1);
    }

    #[test]
    fn test_parse_line_with_only_numbers() {
        let line = parse_line("123 456.78 -9.0");

        assert_eq!(line.tokens.len(), 3);

        match &line.tokens[0] {
            Token::Number((min, max)) => {
                assert_eq!(*min, 123.0);
                assert_eq!(*max, 123.0);
            }
            _ => panic!("expected number token"),
        }

        match &line.tokens[1] {
            Token::Number((min, max)) => {
                assert_eq!(*min, 456.78);
                assert_eq!(*max, 456.78);
            }
            _ => panic!("expected number token"),
        }

        match &line.tokens[2] {
            Token::Number((min, max)) => {
                assert_eq!(*min, -9.0);
                assert_eq!(*max, -9.0);
            }
            _ => panic!("expected number token"),
        }
    }

    #[test]
    fn test_parse_line_with_only_strings() {
        let line = parse_line("hello world test");

        assert_eq!(line.tokens.len(), 3);

        for (i, expected) in ["hello", "world", "test"].iter().enumerate() {
            match &line.tokens[i] {
                Token::String(s) => assert_eq!(s, expected),
                _ => panic!("expected string token"),
            }
        }
    }

    #[test]
    fn test_parse_line_empty_string() {
        let line = parse_line("");
        assert_eq!(line.tokens.len(), 0);
    }

    #[test]
    fn test_aggregation_of_log_lines_with_pids() {
        let mut aggregator = Aggregator::new();

        // Add the provided log lines
        aggregator.add_line("[devvm880.ldc0.facebook.com 2290566] test_actor - INFO - LogMessage from logger: Hey there, from a test!! pid: 2290566").unwrap();
        aggregator.add_line("[devvm880.ldc0.facebook.com 2290555] test_actor - INFO - LogMessage from logger: Hey there, from a test!! pid: 2290555").unwrap();
        aggregator.add_line("[devvm880.ldc0.facebook.com 2290564] test_actor - INFO - LogMessage from logger: Hey there, from a test!! pid: 2290564").unwrap();
        aggregator.add_line("[devvm880.ldc0.facebook.com 2290557] test_actor - INFO - LogMessage from logger: Hey there, from a test!! pid: 2290557").unwrap();

        // Check that we have only one aggregated line
        assert_eq!(aggregator.lines.len(), 1);

        // Check that the count is 4
        assert_eq!(aggregator.lines[0].count, 4);

        // Print the tokens to see how they were parsed
        println!("Tokens: {:?}", aggregator.lines[0].tokens);

        // Check that the PIDs have been aggregated as numbers
        if let Token::Number((min, max)) = &aggregator.lines[0].tokens[1] {
            assert_eq!(*min, 2290555.0); // Min of all PIDs
            assert_eq!(*max, 2290566.0); // Max of all PIDs
        } else {
            panic!("expected number token for PID");
        }

        // Check that the closing bracket was correctly separated
        assert_eq!(
            aggregator.lines[0].tokens[2],
            Token::String("]".to_string())
        );

        // Print the aggregated line to see what it looks like
        println!("Aggregated log line: {}", aggregator);
    }
}
