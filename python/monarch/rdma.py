# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes

import traceback

from dataclasses import dataclass
from traceback import extract_tb, StackSummary
from typing import cast, Dict, Optional, Tuple

import torch

from monarch._rust_bindings.monarch_hyperactor.proc import ActorId

from monarch.actor_mesh import (
    _ActorMeshRefImpl,
    Actor,
    ActorMeshRef,
    endpoint,
    MonarchContext,
)


@dataclass
class LocalRDMARecord:
    data: torch.Tensor


_local_buffers: Dict[int, "LocalRDMARecord"] = {}


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


class RDMAManager(Actor):
    @staticmethod
    def on_proc(proc_id: str) -> "RDMAManager":
        ctx = MonarchContext.get()
        return cast(
            RDMAManager,
            ActorMeshRef(
                RDMAManager,
                _ActorMeshRefImpl.from_actor_id(
                    ctx.mailbox,
                    ActorId.from_string(f"{proc_id}.rdma_manager[0]"),
                ),
                ctx.mailbox,
            ),
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


def _assert_tensor_is_1d_contiguous_uint8(t: torch.Tensor) -> None:
    if t.ndim != 1:
        raise ValueError(f"Tensor must be 1D, got {t.ndim}D")
    if t.dtype != torch.uint8:
        raise ValueError(f"Tensor must be uint8, got {t.dtype}")
    if not t.is_contiguous():
        raise ValueError("Tensor must be contiguous")


class RDMABuffer:
    def __init__(self, data: torch.Tensor) -> None:
        """
        RDMABuffer only supports 1D contiguous tensors that are 1 byte per item.

        To create a 1 byte, 1D view, use t.view(torch.uint8).flatten()

        TODO: Create TensorBuffer, which will be main user API supporting non-contiguous , multi-byte-per-elment tensors
        """
        _assert_tensor_is_1d_contiguous_uint8(data)
        assert data.storage_offset() == 0
        storage = data.untyped_storage()
        self.addr: int = storage.data_ptr()
        self.begin = 0
        self.end: int = storage.size()
        self.proc_id: str = MonarchContext.get().proc_id
        self.local_data: object = None
        _local_buffers[self.addr] = LocalRDMARecord(data)

    def drop(self) -> None:
        if self.proc_id is None:
            del _local_buffers[self.addr]
            return
        rmda_actor = RDMAManager.on_proc(self.proc_id)
        # pyre-ignore[16]: Undefined attribute [16]: `Endpoint` has no attribute `cast`.
        rmda_actor.drop.cast(self.addr)

    def __getstate__(self) -> Tuple[int, int, int, Optional[str]]:
        proc_id = self.proc_id
        # locally created RDMABuffer being set remotely,
        # record its proc_id so we know how to establish connections to it
        if proc_id is None:
            proc_id = MonarchContext.get().proc_id
        return (self.addr, self.begin, self.end, proc_id)

    def __setstate__(self, state: Tuple[int, int, int, str]) -> None:
        self.local_data = None
        self.addr, self.begin, self.end, self.proc_id = state

    async def read_into(self, dst: torch.Tensor, offset: int = 0) -> None:
        """
        Read data from the RDMABuffer into a destination tensor.

        The destination tensor must be contiguous and 1 byte per item.
        """
        _assert_tensor_is_1d_contiguous_uint8(dst)
        bytes = await RDMAManager.on_proc(self.proc_id).fetch.call_one(
            self.addr, offset, dst.numel()
        )
        dst.copy_(torch.frombuffer(bytes, dtype=torch.uint8))

    async def write(self, src: torch.Tensor, offset: int = 0) -> None:
        """
        Write data from a source tensor into the RDMABuffer.

        The source tensor must be contiguous and 1 byte per item.
        """
        _assert_tensor_is_1d_contiguous_uint8(src)
        bytes = _get_bytes(
            src,
            cast(int, src.storage_offset()),
            src.numel(),
        )
        await RDMAManager.on_proc(self.proc_id).put.call_one(self.addr, offset, bytes)


class ActorMeshRefCallFailedException(Exception):
    """
    Deterministic problem with the user's code.
    For example, an OOM resulting in trying to allocate too much GPU memory, or violating
    some invariant enforced by the various APIs.
    """

    def __init__(
        self,
        exception: Exception,
        message: str = "A remote service call has failed asynchronously.",
    ) -> None:
        self.exception = exception
        self.actor_mesh_ref_frames: StackSummary = extract_tb(exception.__traceback__)
        self.message = message

    def __str__(self) -> str:
        exe = str(self.exception)
        actor_mesh_ref_tb = "".join(traceback.format_list(self.actor_mesh_ref_frames))
        return (
            f"{self.message}\n"
            f"Traceback of where the service call failed (most recent call last):\n{actor_mesh_ref_tb}{type(self.exception).__name__}: {exe}"
        )
