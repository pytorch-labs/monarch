# pyre-strict

import argparse
import json
import logging
import os
import pickle
import random
import signal
import sys
import time
from dataclasses import dataclass

from monarch._monarch import (  # manual=//monarch/moanrch_extension:monarch_extension
    hyperactor,
)


logger: logging.Logger = logging.getLogger(__name__)

_ACTOR_NAME = "pingpong"


@dataclass
class PingPongMessage:
    ttl: int


def loop(proc: hyperactor.Proc, iterations: int, batch_size: int) -> None:
    actor = hyperactor.PickledMessageClientActor(proc, _ACTOR_NAME)
    actor_id = actor.actor_id
    peer_actor_id = hyperactor.ActorId(
        world_name=proc.world_name,
        rank=(proc.rank + 1) % 2,
        actor_name=_ACTOR_NAME,
    )
    iter = 0
    ttl = iterations
    while ttl > 0:
        if iter % 2 == proc.rank:
            for _ in range(batch_size):
                ttl -= 1
                msg = PingPongMessage(ttl)
                logger.error(f"sending message with ttl: {ttl}")
                actor.send(
                    peer_actor_id,
                    hyperactor.PickledMessage(
                        sender_actor_id=actor_id,
                        # @lint-ignore PYTHONPICKLEISBAD
                        message=pickle.dumps(msg),
                    ).serialize(),
                )
        else:
            for _ in range(batch_size):
                next_message = actor.get_next_message()
                assert next_message, "No message received"
                # @lint-ignore PYTHONPICKLEISBAD
                msg = pickle.loads(next_message.message)
                ttl = msg.ttl
        iter += 1

    actor.drain_and_stop()


def main() -> int:
    """
    A basic demonstration of Python hyperactor usage. The example consists of two actors
    that ping-pong messages back and forth with a TTL.

    The actors must run on ranks 0 ("ping") and 1 ("pong").

    To run the example, run the following command from fbcode:

    First, start a system that the actors can join:
    $ buck run monarch/hyper -- serve -a [::1]:9000

    Then, start rank 1 (the first receiver):
    $ HYPERACTOR_PROC_ID="ping[1]" HYPERACTOR_BOOTSTRAP_ADDR=tcp![::1]:9000 buck run monarch/hyperactor_python/example:ping_pong_example

    And finally, start rank 0 (the first sender):
    $ HYPERACTOR_PROC_ID="ping[0]" HYPERACTOR_BOOTSTRAP_ADDR=tcp![::1]:9000 buck run monarch/hyperactor_python/example:ping_pong_example

    We currently have to start the actors in this order because otherwise the first
    sender will fail. In the future, we will allow store-and-forward semantics (T207829436)
    that will remove this requirement.
    """

    parser = argparse.ArgumentParser(
        "ping_pong",
    )

    parser.add_argument("-p", "--proc-id", type=str, required=True)
    parser.add_argument("-a", "--boostrap-addr", type=str, required=True)
    parser.add_argument("-i", "--iterations", type=int, default=100)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=10)
    args = parser.parse_args()
    # stop self, will be woken up by parent
    print("SYNC_POINT", flush=True)
    print(f"{args.proc_id}", flush=True)
    os.kill(os.getpid(), signal.SIGSTOP)
    start = time.monotonic()
    loop(
        hyperactor.init_proc(
            proc_id=args.proc_id,
            bootstrap_addr=args.boostrap_addr,
        ),
        iterations=args.iterations,
        batch_size=args.batch_size,
    )

    output = args.output

    print("Tracing environment variables:")
    for key, value in os.environ.items():
        print(f"{key}={value}")

    dur = time.monotonic() - start
    per = dur / args.iterations
    print(f"{dur}s / {args.iterations} = {per}s")
    if output is not None:
        with open(output, "w") as f:
            f.write(
                json.dumps(
                    {
                        "iterations": args.iterations,
                        "duration": dur,
                    }
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
