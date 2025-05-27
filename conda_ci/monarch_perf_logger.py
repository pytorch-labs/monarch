# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import json
import logging
import os
from typing import Any, Dict, Optional

from libfb.py.log import set_simple_logging
from rfe.scubadata.scubadata_py3 import ScubaData

SCUBA_TABLE_NAME = "monarch_perf"

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_metrics_from_file(perf_stats_file: str) -> Dict[str, Any]:
    with open(perf_stats_file, "r") as f:
        perf_stats = json.load(f)
    return perf_stats


def log_to_scuba(
    run_descriptor: str,
    is_monarch: int,
    backend_type: str,
    num_iters: int,
    time_to_first_loss: Optional[float],
    run_id: Optional[str],
) -> None:
    perf_stats_file = os.environ.get("PERF_STATS_FILE", None)
    if perf_stats_file:
        logger.info(f"Reading perf stats from {perf_stats_file=}")
        perf_stats = read_metrics_from_file(perf_stats_file)
        logger.info(f"{perf_stats=}")
        time_per_iter = perf_stats.get("time_per_iter", 0)
    else:
        raise ValueError("PERF_STATS_FILE environment variable is not set")

    sample = ScubaData.Sample()
    sample.addNormalValue("run_descriptor", run_descriptor)
    sample.addIntValue("is_monarch", is_monarch)
    sample.addNormalValue("backend_type", backend_type)
    sample.addIntValue("num_iters", num_iters)
    sample.addDoubleValue("time_to_first_loss", time_to_first_loss)
    sample.addDoubleValue("avg_time_per_iter", time_per_iter)

    # Add few RE specific fields
    # These will be helpful for querying the metrics and reconciling with RE logs

    re_session_id = os.environ.get("SESSION_ID", "")
    re_action_key = os.environ.get("RE_ACTION_KEY", "")
    re_action_digest = os.environ.get("ACTION_DIGEST", "")

    sample.addNormalValue("re_session_id", re_session_id)
    sample.addNormalValue("re_action_key", re_action_key)
    sample.addNormalValue("re_action_digest", re_action_digest)

    # optional columns
    sample.addNormalValue("run_id", run_id)

    logger.info(f"Logging Sample to Scuba: {sample}")
    with ScubaData(SCUBA_TABLE_NAME) as scubadata:
        scubadata.addSample(sample)


def main() -> None:
    set_simple_logging(level=logging.INFO)
    logger.info("Running Monarch Perf Logger")
    parser = argparse.ArgumentParser(
        description="Log Monarch performance data to Scuba"
    )
    parser.add_argument(
        "--run-descriptor", type=str, required=True, help="Run descriptor"
    )
    parser.add_argument(
        "--is-monarch",
        type=int,
        required=True,
        help="Is Monarch Run",
    )
    parser.add_argument("--backend-type", type=str, required=True, help="Backend type")
    parser.add_argument(
        "--num-iters", type=int, required=True, help="Number of iterations"
    )
    parser.add_argument("--run-id", type=str, help="Run ID (optional)")
    parser.add_argument(
        "--time-to-first-loss",
        type=float,
        help="Time to first loss",
    )
    args: argparse.Namespace = parser.parse_args()
    log_to_scuba(
        run_descriptor=args.run_descriptor,
        is_monarch=args.is_monarch,
        backend_type=args.backend_type,
        num_iters=args.num_iters,
        time_to_first_loss=args.time_to_first_loss,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    main()  # pragma: no cover
