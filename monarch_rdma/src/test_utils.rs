/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::thread;
use std::time::Duration;
use std::time::Instant;

use crate::rdma_components::RdmaQueuePair;

// Waits for the completion of an RDMA operation.

// This function polls for the completion of an RDMA operation by repeatedly
// sending a `PollCompletion` message to the specified actor mesh and checking
// the returned work completion status. It continues polling until the operation
// completes or the specified timeout is reached.
pub fn wait_for_completion(qp: &RdmaQueuePair, timeout_secs: u64) -> Result<bool, anyhow::Error> {
    let mut is_completed = false;
    let timeout = Duration::from_secs(timeout_secs);
    let start_time = Instant::now();

    while !is_completed && start_time.elapsed() < timeout {
        match qp.poll_completion() {
            Ok(Some(wc)) => {
                if wc.wr_id() == 1 {
                    is_completed = true;
                }
            }
            Ok(None) => {
                // No completion found, sleep a bit before polling again
                #[allow(clippy::disallowed_methods)]
                thread::sleep(Duration::from_millis(1));
            }
            Err(e) => {
                panic!("Error polling for completion: {}", e);
            }
        }
    }

    Ok(false)
}
