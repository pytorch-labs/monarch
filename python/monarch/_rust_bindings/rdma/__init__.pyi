# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, final, Optional

from monarch._rust_bindings.monarch_hyperactor.tokio import PythonTask

@final
class _RdmaMemoryRegionView:
    def __init__(self, addr: int, size_in_bytes: int) -> None: ...

@final
class _RdmaManager:
    device: str
    def __repr__(self) -> str: ...
    @classmethod
    def create_rdma_manager_blocking(proc_mesh: Any) -> Optional[_RdmaManager]: ...
    @classmethod
    async def create_rdma_manager_nonblocking(
        proc_mesh: Any,
    ) -> Optional[_RdmaManager]: ...

@final
class _RdmaBuffer:
    name: str

    @classmethod
    def create_rdma_buffer_blocking(
        cls, addr: int, size: int, proc_id: str, client: Any
    ) -> _RdmaBuffer: ...
    @classmethod
    def create_rdma_buffer_nonblocking(
        cls, addr: int, size: int, proc_id: str, client: Any
    ) -> PythonTask[Any]: ...
    def drop(self, client: Any) -> PythonTask[None]: ...
    def read_into(
        self,
        addr: int,
        size: int,
        local_proc_id: str,
        client: Any,
        timeout: int,
    ) -> PythonTask[Any]: ...
    def write_from(
        self,
        addr: int,
        size: int,
        local_proc_id: str,
        client: Any,
        timeout: int,
    ) -> PythonTask[Any]: ...
    def __reduce__(self) -> tuple: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def new_from_json(json: str) -> _RdmaBuffer: ...
    @classmethod
    def rdma_supported(cls) -> bool: ...
