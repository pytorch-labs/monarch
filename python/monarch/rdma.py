# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes

import traceback
import warnings

from dataclasses import dataclass
from traceback import extract_tb, StackSummary
from typing import cast, Dict, Optional

import torch

from monarch import ActorFuture as Future
from monarch._rust_bindings.monarch_hyperactor.proc import ActorId
from monarch._rust_bindings.rdma import _RdmaBuffer

from monarch.service import Actor, ActorMeshRef, endpoint, MonarchContext, Service


# RDMARead/WriteTransferWarnings are warnings that are only printed once per process.
class RDMAReadTransferWarning(Warning):
    pass


class RDMAWriteTransferWarning(Warning):
    pass


warnings.simplefilter("once", RDMAReadTransferWarning)
warnings.simplefilter("once", RDMAWriteTransferWarning)


@dataclass
class LocalRDMARecord:
    data: torch.Tensor


def _assert_tensor_is_1d_contiguous_uint8(t: torch.Tensor) -> None:
    if t.ndim != 1:
        raise ValueError(f"Tensor must be 1D, got {t.ndim}D")
    if t.dtype != torch.uint8:
        raise ValueError(f"Tensor must be uint8, got {t.dtype}")
    if not t.is_contiguous():
        raise ValueError("Tensor must be contiguous")


class RDMABuffer:
    def __init__(self, data: torch.Tensor, name: Optional[str] = None) -> None:
        """
        RDMABuffer only supports 1D contiguous tensors that are 1 byte per item.

        To create a 1 byte, 1D view, use t.view(torch.uint8).flatten()

        TODO: Create TensorBuffer, which will be main user API supporting non-contiguous , multi-byte-per-elment tensors
        """
        if name is None:
            name = "rdma_buffer"

        assert _RdmaBuffer.rdma_supported()

        if data.device.type != "cpu":
            raise ValueError(
                "RDMABuffer currently only supports CPU tensors (got device {})".format(
                    data.device
                )
            )

        _assert_tensor_is_1d_contiguous_uint8(data)
        assert data.storage_offset() == 0

        try:
            storage = data.untyped_storage()
            addr: int = storage.data_ptr()
            size = storage.element_size() * data.numel()
            ctx = MonarchContext.get()
            f = Future(
                lambda: _RdmaBuffer.create_rdma_buffer_nonblocking(
                    name=name,
                    addr=addr,
                    size=size,
                    proc_id=ctx.proc_id,
                    client=ctx.mailbox,
                ),
                lambda: _RdmaBuffer.create_rdma_buffer_blocking(
                    name=name,
                    addr=addr,
                    size=size,
                    proc_id=ctx.proc_id,
                    client=ctx.mailbox,
                ),
            )
            self._buffer: _RdmaBuffer = f.get()
        # TODO - specific exception
        except Exception as e:
            print("failed to create buffer ", e)
            raise e

    def drop(self) -> None:
        ctx = MonarchContext.get()
        self._buffer.drop_blocking(ctx.mailbox)

    async def read_into(
        self, dst: torch.Tensor, offset: int = 0, timeout: Optional[int] = 3
    ) -> Optional[int]:
        """
        Read data from the RDMABuffer into a destination tensor.

        The destination tensor must be contiguous and 1 byte per item.

        Note - `timeout` set to None is a valid option, in which case a work_id is returned.
        TODO - The APIs for polling and waiting for the work to complete are not yet exposed.
        """
        _assert_tensor_is_1d_contiguous_uint8(dst)
        dst_gpu = None
        if dst.device.type != "cpu":
            warnings.warn(
                "note: read_into only supports CPU tensors, so `dst` is being copied to CPU.",
                RDMAReadTransferWarning,
            )
            dst_gpu = dst
            dst = dst.cpu()

        storage = dst.untyped_storage()
        addr: int = storage.data_ptr() + offset
        size = storage.element_size() * dst.numel()
        if offset + size > dst.numel():
            raise ValueError(
                f"offset + size ({offset + size }) must be <= dst.numel() ({dst.numel()})"
            )
        ctx = MonarchContext.get()
        res = await self._buffer.read_into(
            addr=addr,
            size=size,
            caller_proc_id=ctx.proc_id,
            client=ctx.mailbox,
            timeout=timeout,
        )
        if dst_gpu is not None:
            dst_gpu.copy_(dst)
        return res

    async def write_from(
        self, src: torch.Tensor, offset: int = 0, timeout: Optional[int] = 3
    ) -> None:
        """
        Write data from a source tensor into the RDMABuffer.

        The source tensor must be contiguous and 1 byte per item.

        Note - `timeout` set to None is a valid option, in which case a work_id is returned.
        TODO - The APIs for polling and waiting for the work to complete are not yet exposed.
        """
        _assert_tensor_is_1d_contiguous_uint8(src)
        if src.device.type != "cpu":
            warnings.warn(
                "note: write_from only supports CPU tensors, so we will write to CPU first, then transfer to `src` in place.",
                RDMAWriteTransferWarning,
            )
            print("warning: write_from only supports CPU tensors.")
            src = src.cpu()
        storage = src.untyped_storage()
        addr: int = storage.data_ptr()
        size = storage.element_size() * src.numel()
        if size + offset > src.numel():
            raise ValueError(
                f"size + offset ({size + offset}) must be <= src.numel() ({src.numel()})"
            )
        ctx = MonarchContext.get()
        res = await self._buffer.write_from(
            addr=addr,
            size=size,
            caller_proc_id=ctx.proc_id,
            client=ctx.mailbox,
            timeout=timeout,
        )
        return res


class ServiceCallFailedException(Exception):
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
        self.service_frames: StackSummary = extract_tb(exception.__traceback__)
        self.message = message

    def __str__(self) -> str:
        exe = str(self.exception)
        service_tb = "".join(traceback.format_list(self.service_frames))
        return (
            f"{self.message}\n"
            f"Traceback of where the service call failed (most recent call last):\n{service_tb}{type(self.exception).__name__}: {exe}"
        )
