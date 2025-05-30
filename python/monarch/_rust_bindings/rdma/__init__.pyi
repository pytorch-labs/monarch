# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, final, Optional

@final
class _RdmaMemoryRegionView:
    def __init__(self, addr: int, size_in_bytes: int) -> None: ...

@final
class _RdmaBuffer:
    name: str

    @classmethod
    def create_rdma_buffer_blocking(
        cls, name: str, addr: int, size: int, proc_id: str, client: Any
    ) -> _RdmaBuffer: ...
    @classmethod
    async def create_rdma_buffer_nonblocking(
        cls, name: str, addr: int, size: int, proc_id: str, client: Any
    ) -> Any: ...
    async def drop(self, client: Any): ...
    def drop_blocking(self, client: Any): ...
    async def read_into(
        self,
        addr: int,
        size: int,
        caller_proc_id: str,
        client: Any,
        timeout: Optional[int],
    ) -> Any: ...
    def read_into_blocking(
        self,
        addr: int,
        size: int,
        caller_proc_id: str,
        client: Any,
        timeout: Optional[int],
    ) -> Any: ...
    async def write_from(
        self,
        addr: int,
        size: int,
        caller_proc_id: str,
        client: Any,
        timeout: Optional[int],
    ) -> Any: ...
    def write_from_blocking(
        self,
        addr: int,
        size: int,
        caller_proc_id: str,
        client: Any,
        timeout: Optional[int],
    ) -> Any: ...
    async def release(
        self, addr: int, size: int, caller_proc_id: str, client: Any
    ) -> Any: ...
    def release_blocking(
        self, addr: int, size: int, caller_proc_id: str, client: Any
    ) -> Any: ...
    def __reduce__(self) -> tuple: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def new_from_json(json: str) -> _RdmaBuffer: ...
    @classmethod
    def rdma_supported(cls) -> bool: ...
