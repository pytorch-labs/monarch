# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import os
import sys
from typing import cast

from monarch._rust_bindings.monarch_hyperactor.channel import ChannelAddr

from monarch._rust_bindings.monarch_hyperactor.debug import (
    debug_cli_client,
    get_external_debug_router_id,
)
from monarch._src.actor.actor_mesh import ActorMesh
from monarch._src.actor.debugger import (
    _get_debug_server_addr,
    DebugCliInput,
    DebugCliOutput,
    DebugCliQuit,
    DebugController,
)


_MONARCH_DEBUG_CLI_ADDR_ENV_VAR = "MONARCH_DEBUG_CLI_ADDR"
_MONARCH_DEBUG_CLI_ADDR_ENV_VAR_DEFAULT = "tcp![::1]:29701"


def run():
    server_addr = _get_debug_server_addr()
    listen_addr_str = os.environ.get(
        _MONARCH_DEBUG_CLI_ADDR_ENV_VAR, _MONARCH_DEBUG_CLI_ADDR_ENV_VAR_DEFAULT
    )
    client = debug_cli_client(server_addr, ChannelAddr.parse(listen_addr_str))
    actor_mesh = cast(
        DebugController,
        ActorMesh.from_actor_id(
            DebugController,
            get_external_debug_router_id(),
            client._mailbox,
        ),
    )

    actor_mesh.enter.call_one(client.actor_id, listen_addr_str).get()
    while True:
        messages = actor_mesh.debug_cli_output.call_one(client.actor_id).get()
        for message in messages:
            match message:
                case DebugCliOutput(data):
                    sys.stdout.write(data)
                    sys.stdout.flush()
                case DebugCliInput(prompt):
                    actor_mesh.debug_cli_input.call_one(
                        input(prompt), client.actor_id
                    ).get()
                case DebugCliQuit():
                    sys.exit(0)
