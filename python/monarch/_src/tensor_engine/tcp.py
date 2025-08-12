# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file provides a TCPBuffer, a light-weight version of RDMABuffer that works on
any hardware.
"""

import ctypes

from dataclasses import dataclass
from typing import cast, Dict, Optional, Tuple

import torch
import zmq
import zmq.asyncio

from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch._src.actor.actor_mesh import Actor, ActorMesh, MonarchContext
from monarch._src.actor.endpoint import endpoint
from monarch._src.actor.future import Future


@dataclass
class LocalTCPRecord:
    data: torch.Tensor


@dataclass
class ZMQConnectionInfo:
    """Connection information needed to establish ZMQ connection"""

    endpoint: str  # ZMQ endpoint (e.g., "tcp://127.0.0.1:5555")


class ZMQConnection:
    """Manages a bidirectional ZMQ connection between two TCPManagers"""

    def __init__(self, context: zmq.asyncio.Context, is_server: bool = False):
        self.context = context
        self.is_server = is_server
        self.send_socket: Optional[zmq.asyncio.Socket] = None
        self.recv_socket: Optional[zmq.asyncio.Socket] = None
        self.endpoint: Optional[str] = None
        self.connected = False

    def initialize(self) -> ZMQConnectionInfo:
        """Initialize the connection and return connection info"""
        if self.is_server:
            # Server creates a PAIR socket and binds to a random port
            self.send_socket = self.context.socket(zmq.PAIR)
            port = self.send_socket.bind_to_random_port("tcp://127.0.0.1")
            self.endpoint = f"tcp://127.0.0.1:{port}"
            self.recv_socket = self.send_socket  # PAIR socket is bidirectional
        else:
            # Client will connect later
            self.send_socket = self.context.socket(zmq.PAIR)
            self.recv_socket = self.send_socket  # PAIR socket is bidirectional

        return ZMQConnectionInfo(endpoint=self.endpoint or "")

    def connect(self, connection_info: ZMQConnectionInfo) -> None:
        """Connect to the remote endpoint"""
        if not self.is_server and self.send_socket:
            self.send_socket.connect(connection_info.endpoint)
            self.endpoint = connection_info.endpoint
        self.connected = True

    async def send(self, data: bytes) -> None:
        """Send data through the connection"""
        if not self.connected or not self.send_socket:
            raise RuntimeError("Connection not established")
        await self.send_socket.send(data)

    async def recv(self) -> bytes | zmq.Frame:
        """Receive data from the connection"""
        if not self.connected or not self.recv_socket:
            raise RuntimeError("Connection not established")
        return await self.recv_socket.recv()

    def close(self) -> None:
        """Close the connection"""
        if self.send_socket:
            self.send_socket.close()
        if self.recv_socket and self.recv_socket != self.send_socket:
            self.recv_socket.close()
        self.connected = False


_local_buffers: Dict[int, "LocalTCPRecord"] = {}


def _get_bytes(storage: torch.Tensor, offset: int, size: int) -> bytearray:
    """Extracts a bytearray from a 1D, 1byte per item tensor."""
    if offset + size > storage.numel():
        raise ValueError(f"Read out of range: {offset + size} > {storage.size()}")
    addr = storage.data_ptr()
    if storage.device.type != "cpu":
        result = bytearray(size)
        result_tensor = torch.frombuffer(
            result,
            dtype=torch.uint8,
        )
        source_tensor = storage[offset:]
        result_tensor.copy_(source_tensor)
    else:
        ctypes_array = (ctypes.c_byte * size).from_address(addr)
        result = bytearray(ctypes_array)
    return result


class TCPManager(Actor):
    # Note - we go through ZMQ instead of Monarch's TCP implementation
    # to bypass Rust limitations we've seen...
    def __init__(self):
        # Map between ActorIds and their corresponding ZMQConnection
        self.connection_map: Dict[ActorId, ZMQConnection] = {}
        # ZMQ context for managing sockets (lazy-initialized)
        self._zmq_context: Optional[zmq.asyncio.Context] = None

    @property
    def zmq_context(self) -> zmq.asyncio.Context:
        """Lazy-initialize ZMQ context to avoid serialization issues"""
        if self._zmq_context is None:
            self._zmq_context = zmq.asyncio.Context()
        return self._zmq_context

    def __reduce__(self) -> Tuple[type, Tuple[()]]:
        """
        Custom pickle reduction that only preserves the class type.
        Similar to how ActorMeshRef handles pickling - we don't serialize
        the ZMQ connections or context, just recreate a fresh TCPManager.
        """
        return (self.__class__, ())

    @staticmethod
    def on_proc(proc_id: str) -> "ActorMesh[TCPManager]":
        ctx = MonarchContext.get()
        return ActorMesh.from_actor_id(
            Class=TCPManager,
            actor_id=ActorId.from_string(f"{proc_id}.tcp_manager[0]"),
            mailbox=ctx.mailbox,
        )

    @endpoint
    async def drop(self, addr: int) -> None:
        if addr in _local_buffers:
            del _local_buffers[addr]

    @endpoint
    async def fetch(self, addr: int, offset: int, nbytes: int) -> bytearray:
        if addr not in _local_buffers:
            raise ValueError(f"Unknown buffer {addr}")
        storage = _local_buffers[addr].data
        return _get_bytes(storage, offset, nbytes)

    @endpoint
    async def put(self, addr: int, offset: int, bytes: bytearray) -> None:
        if addr not in _local_buffers:
            raise ValueError(f"Unknown buffer {addr}")
        storage = _local_buffers[addr].data
        storage[offset : offset + len(bytes)] = torch.frombuffer(
            bytes, dtype=storage.dtype
        )

    def _is_connected(self, other_id: ActorId) -> bool:
        """Check if connected to another TCPManager"""
        if other_id not in self.connection_map:
            return False
        return self.connection_map[other_id].connected

    @endpoint
    def is_connected(self, other_id: ActorId) -> bool:
        """Check if connected to another TCPManager"""
        return self._is_connected(other_id)

    def _initialize_connection(self, remote_id: ActorId) -> bool:
        """Initialize a new ZMQ connection with another TCPManager"""
        if remote_id in self.connection_map:
            return True  # Already initialized

        # Determine if this actor should be the server (based on actor ID comparison)
        current_id = MonarchContext.get().mailbox.actor_id
        is_server = str(current_id) < str(remote_id)

        connection = ZMQConnection(self.zmq_context, is_server=is_server)
        connection.initialize()
        self.connection_map[remote_id] = connection

        return True

    @endpoint
    async def initialize_connection(self, other_id: ActorId) -> bool:
        """Initialize a new ZMQ connection with another TCPManager"""
        return self._initialize_connection(other_id)

    def _connection_info(self, other_id: ActorId) -> ZMQConnectionInfo:
        """Get connection information for establishing a ZMQ connection"""
        if other_id not in self.connection_map:
            raise ValueError(f"No connection initialized for actor {other_id}")

        connection = self.connection_map[other_id]
        if not connection.endpoint:
            raise ValueError(
                f"Connection not properly initialized for actor {other_id}"
            )

        return ZMQConnectionInfo(endpoint=connection.endpoint)

    @endpoint
    async def connection_info(self, other_id: ActorId) -> ZMQConnectionInfo:
        """Get connection information for establishing a ZMQ connection"""
        return self._connection_info(other_id)

    def _connect(self, other_id: ActorId, connection_info: ZMQConnectionInfo) -> None:
        """Establish connection with another TCPManager using provided connection info"""
        if other_id not in self.connection_map:
            raise ValueError(f"No connection initialized for actor {other_id}")

        connection = self.connection_map[other_id]
        connection.connect(connection_info)

    @endpoint
    async def connect(self, other_id: ActorId, connection_info: ZMQConnectionInfo):
        """Establish connection with another TCPManager using provided connection info"""
        self._connect(other_id, connection_info)

    @endpoint
    async def request_connection(self, remote_id: ActorId) -> ZMQConnection:
        """
        Main method to get/create connections with another TCPManager.
        Similar to RDMAManager's request_queue_pair.
        """
        current_id = MonarchContext.get().mailbox.actor_id

        if not self._is_connected(remote_id):
            is_loopback = remote_id == current_id

            if is_loopback:
                self._initialize_connection(remote_id)
                connection_info = self._connection_info(remote_id)
                self._connect(remote_id, connection_info)
            else:
                # Get remote TCPManager reference
                remote_tcp_manager = TCPManager.on_proc(remote_id.proc_id)

                # Initialize connections on both sides
                self._initialize_connection(remote_id)
                # pyre-ignore[16]: Endpoint is not propagating through on_proc.
                await remote_tcp_manager.initialize_connection.call_one(current_id)

                # Exchange connection information
                remote_connection_info = (
                    await remote_tcp_manager.connection_info.call_one(current_id)
                )
                self._connect(remote_id, remote_connection_info)

                local_connection_info = self._connection_info(remote_id)
                await remote_tcp_manager.connect.call_one(
                    current_id, local_connection_info
                )

        connection = self.connection_map.get(remote_id)
        if not connection:
            raise RuntimeError(f"Failed to establish connection with {remote_id}")

        return connection


def _assert_tensor_is_1d_contiguous_uint8(t: torch.Tensor) -> None:
    if t.ndim != 1:
        raise ValueError(f"Tensor must be 1D, got {t.ndim}D")
    if t.dtype != torch.uint8:
        raise ValueError(f"Tensor must be uint8, got {t.dtype}")
    if not t.is_contiguous():
        raise ValueError("Tensor must be contiguous")


class TCPBuffer:
    def __init__(self, data: torch.Tensor) -> None:
        """
        TCPBuffer only supports 1D contiguous tensors that are 1 byte per item.

        To create a 1 byte, 1D view, use t.view(torch.uint8).flatten()
        """
        _assert_tensor_is_1d_contiguous_uint8(data)
        assert data.storage_offset() == 0
        storage = data.untyped_storage()
        self.addr: int = storage.data_ptr()
        self.begin = 0
        self.end: int = storage.size()
        self.proc_id: str = MonarchContext.get().proc_id
        self.local_data: object = None
        _local_buffers[self.addr] = LocalTCPRecord(data)

    def drop(self) -> None:
        if self.proc_id is None:
            del _local_buffers[self.addr]
            return
        rmda_actor = TCPManager.on_proc(self.proc_id)
        # pyre-ignore[16]: Undefined attribute [16]: `Endpoint` has no attribute `cast`.
        rmda_actor.drop.cast(self.addr)

    def __getstate__(self) -> Tuple[int, int, int, Optional[str]]:
        proc_id = self.proc_id
        # locally created TCPBuffer being set remotely,
        # record its proc_id so we know how to establish connections to it
        if proc_id is None:
            proc_id = MonarchContext.get().proc_id
        return (self.addr, self.begin, self.end, proc_id)

    def __setstate__(self, state: Tuple[int, int, int, str]) -> None:
        self.local_data = None
        self.addr, self.begin, self.end, self.proc_id = state

    def read_into(
        self, dst: torch.Tensor, offset: int = 0, *args, **kwargs
    ) -> Future[None]:
        """
        Read data from the TCPBuffer into a destination tensor.

        The destination tensor must be contiguous and 1 byte per item.
        """
        try:
            MonarchContext.get()
        except LookupError:
            raise RuntimeError(
                "TCPBuffer.read_into() can only be called from within a Monarch actor context. "
                "Make sure you're calling this from within an actor method."
            )

        _assert_tensor_is_1d_contiguous_uint8(dst)

        # pyre-ignore[16]: Endpoint is not propagating through on_proc.
        bytes_future = TCPManager.on_proc(self.proc_id).fetch.call_one(
            self.addr, offset, dst.numel()
        )

        async def coro() -> None:
            bytes_ = await bytes_future
            dst.copy_(torch.frombuffer(bytes_, dtype=torch.uint8))

        return Future(coro=coro())

    def write_from(
        self, src: torch.Tensor, offset: int = 0, *args, **kwargs
    ) -> Future[None]:
        """
        Write data from a source tensor into the TCPBuffer.

        The source tensor must be contiguous and 1 byte per item.
        """
        # Check if we're in a Monarch context
        try:
            MonarchContext.get()
        except LookupError:
            raise RuntimeError(
                "TCPBuffer.write_from() can only be called from within a Monarch actor context. "
                "Make sure you're calling this from within an actor method."
            )

        _assert_tensor_is_1d_contiguous_uint8(src)
        bytes_ = _get_bytes(
            src,
            cast(int, src.storage_offset()),
            src.numel(),
        )
        # pyre-ignore[16]: Endpoint is not propagating through on_proc.
        return TCPManager.on_proc(self.proc_id).put.call_one(self.addr, offset, bytes_)
