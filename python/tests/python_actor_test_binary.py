# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import logging
import sys

import click

from monarch.actor import Actor, endpoint, proc_mesh


@click.group()
def main() -> None:
    pass


class Printer(Actor):
    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger()

    @endpoint
    async def print(self, content: str) -> None:
        print(f"{content}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()


async def _flush_logs() -> None:
    pm = await proc_mesh(gpus=2)
    # never flush
    await pm.logging_option(aggregate_window_sec=1000)
    am = await pm.spawn("printer", Printer)

    # These should be streamed to client
    for _ in range(5):
        await am.print.call("has print streaming")

    # Sleep a tiny so we allow the logs to stream back to the client
    await asyncio.sleep(1)


@main.command("flush-logs")
def flush_logs() -> None:
    asyncio.run(_flush_logs())


if __name__ == "__main__":
    main()
