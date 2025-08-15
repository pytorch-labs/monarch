/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs::OpenOptions;
use std::os::fd::BorrowedFd;
use std::os::unix::io::AsRawFd;

use anyhow::Context;
use nix::errno::Errno;
use nix::libc::STDERR_FILENO;
use nix::libc::STDOUT_FILENO;
use nix::unistd::write;

/// Checks if stdout is broken (e.g., due to parent process death).
///
/// Attempts a minimal write to stdout to detect if the pipe is
/// broken. Returns true if stdout is unavailable (broken pipe or bad
/// file descriptor).
pub(crate) fn is_stdout_broken() -> bool {
    // SAFETY: `STDOUT_FILENO` (`1`) is a valid file descriptor by
    // definition. `BorrowedFd::borrow_raw` is safe here because we're
    // only using it for the duration of this function call and not
    // storing it.
    let fd = unsafe { BorrowedFd::borrow_raw(STDOUT_FILENO) };
    matches!(write(fd, b"\0"), Err(Errno::EPIPE | Errno::EBADF))
}

/// Redirects stdout and stderr to the specified file.
///
/// The file is opened in append mode and created if it doesn't exist.
/// This permanently modifies the process's stdio streams.
pub(crate) fn redirect_stdio_to_file(path: &str) -> anyhow::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open log file: {}", path))?;
    let raw_fd = file.as_raw_fd();
    // SAFETY: `raw_fd` is a valid file descriptor obtained from
    // `as_raw_fd()` on an open file. `STDOUT_FILENO` (`1`) and
    // `STDERR_FILENO` (`2`) are always valid file descriptor numbers.
    // `dup2` is safe to call with these valid file descriptors.
    unsafe {
        if nix::libc::dup2(raw_fd, STDOUT_FILENO) == -1 {
            anyhow::bail!(
                "failed to redirect stdout: {}",
                std::io::Error::last_os_error()
            );
        }
        if nix::libc::dup2(raw_fd, STDERR_FILENO) == -1 {
            anyhow::bail!(
                "failed to redirect stderr: {}",
                std::io::Error::last_os_error()
            );
        }
    }
    std::mem::forget(file);
    Ok(())
}

/// Redirects stdout and stderr to a user-specific log file in /tmp.
///
/// Creates a log file at `/tmp/{user}/monarch-process-exit-{pid}.log`
/// and redirects stdio to it. The user directory is created if it
/// doesn't exist.
pub(crate) fn redirect_stdio_to_user_pid_file() -> anyhow::Result<()> {
    let user = std::env::var("USER").unwrap_or_else(|_| "unknown".to_string());
    let pid = std::process::id();
    let log_dir = format!("/tmp/{}", user);
    std::fs::create_dir_all(&log_dir)?;
    let path = format!("{}/monarch-process-exit-{}.log", log_dir, pid);
    redirect_stdio_to_file(&path)?;
    Ok(())
}

/// Redirects stdio to a log file if stdout is broken.
pub(crate) fn handle_broken_pipes() {
    if is_stdout_broken() {
        if redirect_stdio_to_user_pid_file().is_ok() {
            tracing::info!(
                "stdio for {} redirected due to broken pipe",
                std::process::id()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use nix::libc::STDERR_FILENO;
    use nix::libc::STDOUT_FILENO;
    use tempfile::TempDir;

    use super::*;

    struct StdioGuard {
        saved_stdout: i32,
        saved_stderr: i32,
    }

    impl StdioGuard {
        fn new() -> Self {
            // SAFETY: `STDOUT_FILENO` (`1`) and `STDERR_FILENO` (`2`)
            // are always valid file descriptor numbers. `dup()` is
            // safe to call on these standard descriptors and will
            // return new file descriptors pointing to the same files.
            unsafe {
                let saved_stdout = nix::libc::dup(STDOUT_FILENO);
                let saved_stderr = nix::libc::dup(STDERR_FILENO);
                Self {
                    saved_stdout,
                    saved_stderr,
                }
            }
        }
    }

    impl Drop for StdioGuard {
        fn drop(&mut self) {
            // SAFETY: `saved_stdout` and `saved_stderr` are valid
            // file descriptors returned by `dup()` in `new()`.
            // `STDOUT_FILENO` and `STDERR_FILENO` are always valid
            // target descriptors. `dup2()` and `close()` are safe to
            // call with these valid fds.
            unsafe {
                nix::libc::dup2(self.saved_stdout, STDOUT_FILENO);
                nix::libc::dup2(self.saved_stderr, STDERR_FILENO);
                nix::libc::close(self.saved_stdout);
                nix::libc::close(self.saved_stderr);
            }
        }
    }

    #[test]
    fn test_is_stdout_broken_with_working_stdout() {
        assert!(!is_stdout_broken());
    }

    #[test]
    fn test_redirect_stdio_to_file_creates_file() {
        let _guard = StdioGuard::new();

        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("test.log");
        let path_str = log_path.to_str().unwrap();

        assert!(redirect_stdio_to_file(path_str).is_ok());
        assert!(log_path.exists());
    }
}
