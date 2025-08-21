/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! This module implements a cancellation-safe zero-copy framer for network channels.

use std::io;
use std::io::IoSlice;
use std::mem::take;

use bytes::Buf;
use bytes::BufMut;
use bytes::Bytes;
use bytes::BytesMut;
use tokio::io::AsyncRead;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWrite;
use tokio::io::AsyncWriteExt;

/// A FrameReader reads frames from an underlying [`AsyncRead`].
pub struct FrameReader<R> {
    reader: R,
    max_frame_length: usize,
    state: FrameReaderState,
}

enum FrameReaderState {
    /// Accumulating 8-byte length prefix.
    ReadLen { buf: BytesMut }, // buf.len() <= 8
    /// Accumulating body of exactly `len` bytes.
    ReadBody { len: usize, buf: BytesMut }, // buf.len() <= len
}

impl<R: AsyncRead + Unpin> FrameReader<R> {
    /// Create a new framer for `reader`. Frames exceeding `max_frame_length`
    /// in length result in an irrecoverable reader error.
    pub fn new(reader: R, max_frame_length: usize) -> Self {
        Self {
            reader,
            max_frame_length,
            state: FrameReaderState::ReadLen {
                buf: BytesMut::with_capacity(8),
            },
        }
    }

    /// Read the next frame from the underlying reader. If the frame
    /// exceeds the configured maximum length, `next` returns an
    /// `io::ErrorKind::InvalidData` error.
    ///
    /// The method is cancellation safe in the sense that, if it is
    /// used in a branch of a `tokio::select!` block, frames are never
    /// dropped.
    ///
    /// # Errors
    ///
    /// * Returns `io::ErrorKind::InvalidData` if a frame exceeds
    ///   `max_frame_length`. **This error is fatal:** once returned,
    ///   the `FrameReader` must be dropped; the underlying connection
    ///   is no longer valid.
    pub async fn next(&mut self) -> io::Result<Option<Bytes>> {
        loop {
            match &mut self.state {
                FrameReaderState::ReadLen { buf } if buf.len() < 8 => {
                    let n = self.reader.read_buf(buf).await?;

                    // https://docs.rs/tokio/latest/tokio/io/trait.AsyncReadExt.html#method.read_buf
                    // "This reader has reached its “end of file” and will likely no longer
                    // be able to produce bytes. Note that this does not mean that the reader
                    // will always no longer be able to produce bytes."
                    //
                    // In practice, this means EOF.
                    if n == 0 {
                        if buf.is_empty() {
                            // We ended on a frame boundary. End of stream:
                            return Ok(None);
                        } else {
                            return Err(io::ErrorKind::UnexpectedEof.into());
                        }
                    }
                }

                FrameReaderState::ReadLen { buf } => {
                    let len = buf.get_u64() as usize;
                    if len > self.max_frame_length {
                        return Err(io::ErrorKind::InvalidData.into());
                    }
                    self.state = FrameReaderState::ReadBody {
                        len,
                        buf: BytesMut::with_capacity(len),
                    };
                }

                FrameReaderState::ReadBody { len, buf } if buf.len() < *len => {
                    let n = self.reader.read_buf(buf).await?;
                    if n == 0 {
                        return Err(io::ErrorKind::UnexpectedEof.into());
                    }
                }

                FrameReaderState::ReadBody { len, buf } if buf.len() == *len => {
                    let frame = take(buf).freeze();
                    self.state = FrameReaderState::ReadLen {
                        buf: BytesMut::with_capacity(8),
                    };
                    return Ok(Some(frame));
                }
                _ => panic!("impossible state"),
            }
        }
    }
}

/// A Writer for message frames. `FrameWrite` requires the user to drive
/// the underlying state machines through (possibly) successive calls to
/// `send`, retaining cancellation safety. The `FrameWrite` owns the underlying
/// writer until the frame has been written to completion.
pub struct FrameWrite<W, B> {
    writer: W,
    len_buf: Bytes,
    body: B,
}

impl<W: AsyncWrite + Unpin, B: Buf> FrameWrite<W, B> {
    /// Create a new frame writer, writing `body` to `writer`.
    pub fn new(writer: W, body: B) -> Self {
        let mut len_buf = BytesMut::with_capacity(8);
        len_buf.put_u64(body.remaining() as u64);
        let len_buf = len_buf.freeze();
        Self {
            writer,
            len_buf,
            body,
        }
    }

    /// Drive the underlying state machine. The frame is written when this
    /// `send` returns successfully.
    ///
    /// This method is cancellation safe in the sense that each invocation to `send`
    /// preserves progress (the future can be safely dropped at any time). Thus, the
    /// user can drive the state machine by calling `send` multiple times, dropping the
    /// returned futures at any time. Upon completion, the frame is guaranteed to be
    /// written, unless an error was encountered, in which case the underlying stream
    /// is in an undefined state.
    pub async fn send(&mut self) -> io::Result<()> {
        loop {
            if self.len_buf.has_remaining() {
                self.writer.write_all_buf(&mut self.len_buf).await?;
            } else if self.body.has_remaining() {
                // This is safe. According to the docs for write_vectored:
                //
                // > This method is cancellation safe in the sense that if it is used as
                // > the event in a [`tokio::select!`](crate::select) statement and some
                // > other branch completes first, then it is guaranteed that no data was
                // > written to this `AsyncWrite`.
                //
                // We write at most 4 chunks at a time (to be tuned). We may also consider
                // using MaybeUninit here to avoid initialization overhead.
                let mut chunks = [
                    IoSlice::new(&[]),
                    IoSlice::new(&[]),
                    IoSlice::new(&[]),
                    IoSlice::new(&[]),
                ];
                let num_chunks = self.body.chunks_vectored(&mut chunks);
                let count = self.writer.write_vectored(&chunks[0..num_chunks]).await?;
                self.body.advance(count);
            } else {
                return Ok(());
            }
        }
    }

    /// Complete the write, returning ownership of the underlying writer.
    /// This should only be called after successful sends; at other times
    /// the underlying stream is in an undefined state.
    pub fn complete(self) -> W {
        let Self { writer, .. } = self;
        writer
    }

    /// Writes a single frame into the underlying writer and returns
    /// it.
    ///
    /// This is a convenience for the common pattern:
    /// `FrameWrite::new(writer, bytes).send().await?.complete()`.
    ///
    /// Frame writes are atomic: either the entire frame is sent, or
    /// an error is returned. No partial frames are observed by the
    /// receiver.
    ///
    /// # Arguments
    ///
    /// * `writer` — the `AsyncWrite` sink to write into.
    /// * `bytes` — the serialized frame body to send.
    ///
    /// # Returns
    ///
    /// On success, returns the underlying writer so the caller can
    /// continue using it for further frames. On error, returns the
    /// I/O error from the underlying write.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bytes::Bytes;
    ///
    /// // `writer` is any AsyncWrite + Unpin (e.g. a tokio `WriteHalf`)
    /// let writer = FrameWrite::write_frame(writer, Bytes::from_static(b"hello")).await?;
    /// ```
    pub async fn write_frame(writer: W, buf: B) -> std::io::Result<W> {
        let mut fw = FrameWrite::new(writer, buf);
        fw.send().await?;
        Ok(fw.complete())
    }
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::task::Context;
    use std::task::Poll;

    use bytes::Bytes;
    use rand::Rng;
    use rand::thread_rng;
    use tokio::io::AsyncWrite;
    use tokio::io::AsyncWriteExt;

    use super::*;

    fn random_buffer(max_len: usize) -> Bytes {
        let mut rng = thread_rng();
        let len = rng.gen_range(0..max_len);
        let mut buf = vec![0u8; len];
        rng.fill(buf.as_mut_slice());
        Bytes::from(buf)
    }

    #[tokio::test]
    async fn test_framer_roundtrip() {
        const MAX_LEN: usize = 1024;

        let (reader, writer) = tokio::io::duplex(MAX_LEN + 8);
        let mut reader = FrameReader::new(reader, MAX_LEN);

        let mut writer = Some(writer);

        for _ in 0..1024 {
            let body = random_buffer(MAX_LEN);
            let mut frame_write = FrameWrite::new(writer.take().unwrap(), body.clone());
            frame_write.send().await.unwrap();
            writer = Some(frame_write.complete());

            let frame = reader.next().await.unwrap().unwrap();
            assert_eq!(frame, body);
        }
    }

    #[tokio::test]
    async fn test_write_frame_smoke() {
        let (a, b) = tokio::io::duplex(4096);
        let (r, _w_unused) = tokio::io::split(a);
        let (_r_unused, w) = tokio::io::split(b);

        let mut reader = FrameReader::new(r, 1024);

        eprintln!("write 1");
        let w = FrameWrite::write_frame(w, Bytes::from_static(b"hello"))
            .await
            .unwrap();
        eprintln!("write 2");
        let _ = FrameWrite::write_frame(w, Bytes::from_static(b"world"))
            .await
            .unwrap();

        eprintln!("read 1");
        let f1 = reader.next().await.unwrap().unwrap();
        eprintln!("read 2");
        let f2 = reader.next().await.unwrap().unwrap();

        assert_eq!(f1.as_ref(), b"hello");
        assert_eq!(f2.as_ref(), b"world");
    }

    #[tokio::test]
    async fn test_reader_eof_at_boundary() {
        let (a, b) = tokio::io::duplex(4096);
        let (r, _wu) = tokio::io::split(a);
        let (_ru, mut w) = tokio::io::split(b);
        let mut reader = FrameReader::new(r, 1024);

        // Write a complete frame.
        w = FrameWrite::write_frame(w, Bytes::from_static(b"done"))
            .await
            .unwrap();
        // Now, shutdown the writer so the peer gets an EOF.
        w.shutdown().await.unwrap();
        drop(w);
        assert_eq!(
            reader.next().await.unwrap(),
            Some(Bytes::from_static(b"done"))
        );
        // Boundary EOF.
        assert!(reader.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_reader_eof_mid_frame() {
        let (a, b) = tokio::io::duplex(4096);
        let (r, _wu) = tokio::io::split(a);
        let (_ru, mut w) = tokio::io::split(b);
        let mut reader = FrameReader::new(r, 1024);

        // Start a frame of length 5.
        let mut len = bytes::BytesMut::with_capacity(8);
        len.put_u64(5);
        w.write_all(&len.freeze()).await.unwrap();
        // Write only 2 bytes of the body.
        w.write_all(b"he").await.unwrap();
        // Shutdown the writer so there's an EOF mid frame.
        w.shutdown().await.unwrap();

        // Reading back the frame will manifest an error.
        let err = reader.next().await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    /// A wrapper around an `AsyncWrite` that throttles how many bytes
    /// may be written per poll.
    ///
    /// We are going to use this to simulate partial writes to test
    /// cancellation safety: when the budget is 0, `poll_write`
    /// returns `Poll::Pending` and calls the waker so the task is
    /// scheduled to be polled again later.
    struct Throttled<W> {
        inner: W,
        // Number of bytes allowed to be written in the next poll. If
        // 0, writes return `Poll::Pending`.
        budget: usize,
    }

    impl<W> Throttled<W> {
        fn new(inner: W) -> Self {
            Self {
                inner,
                budget: usize::MAX,
            }
        }

        fn set_budget(&mut self, n: usize) {
            self.budget = n;
        }
    }

    impl<W: AsyncWrite + Unpin> AsyncWrite for Throttled<W> {
        fn poll_write(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            // No budget left this poll. Return "not ready" and ask to
            // be polled again later.
            if self.budget == 0 {
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
            let n = buf.len().min(self.budget);
            self.budget -= n;
            // Delegate a write of the first `n` bytes to the inner
            // writer.
            Pin::new(&mut self.inner).poll_write(cx, &buf[..n])
        }

        fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            // Delegate to `inner` for flushing.
            Pin::new(&mut self.inner).poll_flush(cx)
        }

        fn poll_shutdown(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            // Delegate to `inner` (ensure resources are released and
            // `EOF` is signaled downstream).
            Pin::new(&mut self.inner).poll_shutdown(cx)
        }
    }

    #[tokio::test]
    #[allow(clippy::disallowed_methods)]
    async fn test_writer_cancellation_resume() {
        let (a, b) = tokio::io::duplex(4096);
        let (r, _wu) = tokio::io::split(a);
        let (_ru, w) = tokio::io::split(b);

        let w = Throttled::new(w);
        // 256 bytes, all = 0x2A ('*'), "the answer"
        let body = Bytes::from_static(&[42u8; 256]);
        let mut reader = FrameReader::new(r, 1024 * 1024);
        let mut fw = FrameWrite::new(w, body.clone());

        // Allow only the 8-byte length to be written, then cancel.
        fw.writer.set_budget(8);
        let fut = fw.send();
        tokio::select! {
            _ = fut => panic!("send unexpectedly completed"),
            _ = tokio::time::sleep(std::time::Duration::from_millis(5)) => {}
        }
        // The `fut` is dropped here i.e. "cancellation".
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), async {
                reader.next().await
            })
            .await
            .is_err(),
            "a full frame isn't available yet, so reader.next().await should block"
        );

        // Now allow the remaining body to flush and complete the
        // frame.
        fw.writer.set_budget(usize::MAX);
        fw.send().await.unwrap();
        let mut w = fw.complete();
        let got = reader.next().await.unwrap().unwrap();
        assert_eq!(got, body);

        // Shutdown and test for EOF on boundary.
        w.shutdown().await.unwrap();
        assert!(reader.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_reader_accepts_exact_max_len_frames() {
        const MAX: usize = 1024;
        const BUFSIZ: usize = 8 + MAX; // BUFSIZ (bytes) = 8 (len) + MAX (body)
        let (a, b) = tokio::io::duplex(BUFSIZ);
        let (r, _wu) = tokio::io::split(a);
        let (_ru, mut w) = tokio::io::split(b);
        let mut reader = FrameReader::new(r, MAX);

        let bytes_written = Bytes::from(vec![0xAB; MAX]);
        w = FrameWrite::write_frame(w, bytes_written.clone())
            .await
            .unwrap();

        let bytes_read = reader.next().await.unwrap().unwrap();
        assert_eq!(bytes_read.len(), MAX);
        assert_eq!(bytes_read, bytes_written);

        w.shutdown().await.unwrap();
        assert!(reader.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_reader_rejects_over_max_len_frames() {
        const MAX: usize = 1024;
        const BUFSIZ: usize = 8 + MAX; // BUFSIZ (bytes) = 8 (len) + MAX (body)
        let (a, b) = tokio::io::duplex(BUFSIZ);
        let (r, _wu) = tokio::io::split(a);
        let (_ru, mut w) = tokio::io::split(b);
        let mut reader = FrameReader::new(r, MAX - 1);

        let bytes_written = Bytes::from(vec![0xAB; MAX]);
        w = FrameWrite::write_frame(w, bytes_written).await.unwrap();

        let err = reader.next().await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);

        // Do NOT try to use `reader` beyond this point! There has
        // been a protocol violation: `InvalidData` means the stream
        // is corrupted and the only valid thing you can do with it is
        // `drop` it.
        drop(reader);

        w.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_reader_accepts_zero_len_frames() {
        const MAX: usize = 0;
        const BUFSIZ: usize = 8 + MAX; // BUFSIZ (bytes) = 8 (len) + MAX (body)
        let (a, b) = tokio::io::duplex(BUFSIZ);
        let (r, _wu) = tokio::io::split(a);
        let (_ru, mut w) = tokio::io::split(b);
        let mut reader = FrameReader::new(r, MAX);

        w = FrameWrite::write_frame(w, Bytes::new()).await.unwrap();
        assert_eq!(reader.next().await.unwrap().unwrap().len(), 0);
        w = FrameWrite::write_frame(w, Bytes::new()).await.unwrap();
        assert_eq!(reader.next().await.unwrap().unwrap().len(), 0);

        w.shutdown().await.unwrap();
        assert!(reader.next().await.unwrap().is_none());
    }
}
