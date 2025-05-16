#![allow(dead_code)] // until used

use std::mem::swap;
use std::mem::take;
use std::ops::DerefMut;
use std::sync::Arc;
use std::sync::Mutex;

use tokio::io;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncRead;
use tokio::io::AsyncWrite;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;

/// A tailer (ring buffer) of (text) log lines.
pub struct LogTailer {
    state: Arc<Mutex<State>>,
    handle: tokio::task::JoinHandle<Result<(), std::io::Error>>,
}

#[derive(Clone, Default)]
struct State {
    next: usize,
    lines: Vec<String>,
}

impl LogTailer {
    /// Create a new tailer given a `stream`. The tailer tails the reader in the
    /// background, while keeping at most `max` log lines in its buffer. The tailer
    /// stops when the stream is ended (i.e., returns an EOF).
    pub fn new(max: usize, stream: impl AsyncRead + Send + Unpin + 'static) -> Self {
        Self::tee(max, stream, io::sink())
    }

    /// Create a new tailer given a `stream`. The tailer tails the reader in the
    /// background, while keeping at most `max` log lines in its buffer. The tailer
    /// stops when the stream is ended (i.e., returns an EOF). All lines read by the
    /// tailer are teed onto the provided `tee` stream.
    pub fn tee(
        max: usize,
        stream: impl AsyncRead + Send + Unpin + 'static,
        mut tee: impl AsyncWrite + Send + Unpin + 'static,
    ) -> Self {
        let state = Arc::new(Mutex::new(State {
            next: 0,
            lines: Vec::with_capacity(max),
        }));
        let cloned_state = Arc::clone(&state);

        // todo: handle error case, stuff the handle here,
        // and make this awaitable, etc
        let handle = tokio::spawn(async move {
            let mut reader = BufReader::new(stream);
            let mut buffer = String::new();
            loop {
                buffer.clear(); // clear retains the buffer
                // TODO: we should probably limit line length
                if reader.read_line(&mut buffer).await? == 0 {
                    break Ok(());
                }
                let _ = tee.write_all(buffer.as_bytes()).await;
                while buffer.ends_with('\n') {
                    buffer.pop();
                }
                let mut locked = state.lock().unwrap();
                let next = locked.next;
                if next < locked.lines.len() {
                    swap(&mut locked.lines[next], &mut buffer);
                } else {
                    locked.lines.push(buffer.clone());
                }
                locked.next = (next + 1) % max;
            }
        });

        LogTailer {
            state: cloned_state,
            handle,
        }
    }

    /// Get the last log lines in the buffer. Returns up to `max` lines.
    pub fn tail(&self) -> Vec<String> {
        let State { next, mut lines } = self.state.lock().unwrap().clone();
        lines.rotate_left(next);
        lines
    }

    /// Abort the tailer. This will stop any ongoing reads, and drop the
    /// stream. Abort is complete after `join` returns.
    pub fn abort(&self) {
        self.handle.abort()
    }

    /// Join the tailer. This waits for the internal tailing task to complete,
    /// and then returns the contents of the line buffer and the status of the
    /// tailer task.
    pub async fn join(self) -> (Vec<String>, Result<(), anyhow::Error>) {
        let result = match self.handle.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        };

        let State { next, mut lines } = take(self.state.lock().unwrap().deref_mut());
        lines.rotate_left(next);
        (lines, result)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use tokio::io::AsyncWriteExt;

    use super::*;

    #[tokio::test]
    async fn test_basic() {
        let reader = Cursor::new("hello\nworld\n".as_bytes());
        let (lines, result) = LogTailer::new(5, reader).join().await;
        assert!(result.is_ok());
        assert_eq!(lines, vec!["hello".to_string(), "world".to_string()]);
    }

    #[tokio::test]
    async fn test_tee() {
        let reader = Cursor::new("hello\nworld\n".as_bytes());
        let (write, read) = io::duplex(64); // 64-byte internal buffer
        let (lines, result) = LogTailer::tee(5, reader, write).join().await;
        assert!(result.is_ok());
        assert_eq!(lines, vec!["hello".to_string(), "world".to_string()]);
        let mut lines = BufReader::new(read).lines();
        assert_eq!(lines.next_line().await.unwrap().unwrap(), "hello");
        assert_eq!(lines.next_line().await.unwrap().unwrap(), "world");
    }

    #[tokio::test]
    async fn test_streaming_logtailer() {
        let (reader, mut writer) = tokio::io::simplex(1);

        let writer_handle = tokio::spawn(async move {
            for i in 1.. {
                writer.write_all(i.to_string().as_bytes()).await.unwrap();
                writer.write_all("\n".as_bytes()).await.unwrap();
            }
        });

        let max_lines = 5;
        let tailer = LogTailer::new(max_lines, reader);

        let target = 1000; // require at least 1000 lines
        let mut last = 0;
        while last < target {
            tokio::task::yield_now().await;
            let lines: Vec<_> = tailer
                .tail()
                .into_iter()
                .map(|line| line.parse::<usize>().unwrap())
                .collect();
            if lines.is_empty() {
                continue;
            }

            assert!(lines.len() <= max_lines);
            assert!(lines[0] > last);
            for i in 1..lines.len() {
                assert_eq!(lines[i], lines[i - 1] + 1);
            }
            last = lines[lines.len() - 1];
        }
        // Unfortunately, there is no way to close the simplex stream
        // from the write half only, so we just have to let this go.
        writer_handle.abort();
        let _ = writer_handle.await;
        tailer.abort();
        tailer.join().await.1.unwrap_err();
    }
}
