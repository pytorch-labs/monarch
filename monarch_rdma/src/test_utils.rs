/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::time::Duration;
use std::time::Instant;

use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;

use crate::rdma_components::RdmaQueuePair;

// Waits for the completion of an RDMA operation.

// This function polls for the completion of an RDMA operation by repeatedly
// sending a `PollCompletion` message to the specified actor mesh and checking
// the returned work completion status. It continues polling until the operation
// completes or the specified timeout is reached.

#[allow(dead_code)]
pub async fn wait_for_completion(
    qp: &RdmaQueuePair,
    timeout_secs: u64,
) -> Result<bool, anyhow::Error> {
    let timeout = Duration::from_secs(timeout_secs);
    let start_time = Instant::now();

    while start_time.elapsed() < timeout {
        match qp.poll_completion() {
            Ok(Some(wc)) => {
                if wc.wr_id() == 0 {
                    return Ok(true);
                }
            }
            Ok(None) => {
                RealClock.sleep(Duration::from_millis(1)).await;
            }
            Err(e) => {
                return Err(anyhow::anyhow!(e));
            }
        }
    }

    Ok(false)
}
