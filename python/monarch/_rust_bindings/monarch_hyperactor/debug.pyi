# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import TYPE_CHECKING

from monarch._rust_bindings.monarch_hyperactor.channel import ChannelAddr
from monarch._rust_bindings.monarch_hyperactor.mailbox import Mailbox
from monarch._rust_bindings.monarch_hyperactor.proc import ActorId

if TYPE_CHECKING:
    from monarch._src.actor.actor_mesh import Instance  # type: ignore

def debug_cli_client(server_addr: ChannelAddr, listen_addr: ChannelAddr) -> "Instance":  # type: ignore
    """
    Gets a clone of the debug CLI's instance, used for communicating with the debugee process.

    Args:
        server_addr: The channel address of the debug server on the debugee process.
        listen_addr: The address on which to listen for responses from the debug server.

    Returns:
        An Instance object for debug CLI communication
    """
    ...

def get_external_debug_router_id() -> ActorId:
    """
    Retrieves the actor ID that external callers can send messages to on the debugee process
    such that they are routed to the main debug actor.

    Returns:
        ActorId for communicating with the debugee process
    """
    ...

def bind_debug_cli_actor(
    debug_cli_actor_id: ActorId, response_addr: ChannelAddr
) -> None:
    """
    Binds a debug CLI actor to a response address.

    Args:
        debug_cli_actor_id: The ActorId of the debug CLI actor
        response_addr: The address to send responses to
    """
    ...

def init_debug_server(
    debug_controller_mailbox: Mailbox, listen_addr: ChannelAddr
) -> None:
    """
    Initializes the debug server on the debugee process.

    Args:
        debug_controller_mailbox: The mailbox for the debug controller actor on the debugee process
        listen_addr: The address for the debug server to listen on
    """
    ...
