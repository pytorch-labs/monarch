# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import uuid
from collections import deque
from typing import Any, Dict

import torch

from . import device_mesh, messages, stream
from .fake import fake_call
from .function import ResolvableFunctionFromPath
from .reference import Referenceable
from .remote import Remote, remote
from .tensor import dtensor_check, Tensor
from .tree import flatten


def remote_generator(path: str, max_messages: int = 50):
    def wrapper(annotation):
        fn = remote(path, propagate=annotation)
        return lambda *args, **kwargs: create_pipe(
            fn, *args, max_messages=max_messages, **kwargs
        )

    return wrapper


def create_pipe(fn, *args, max_messages: int = 50, **kwargs):
    return Pipe(fn, max_messages, args, kwargs)


class Pipe(Referenceable):
    """
    Pipe abstraction on the controller. Designed to to be used with ipc PAIR sockets, e.g dataloaders and trainers.

    Example::
        @remote_generator('dataloader.main')
        def dataloader_pipe(pipe: Pipe, batch_size: int, sequence_length: int):
            while True:
                yield {
                    'input': torch.zeros(batch_size, sequence_length),
                    'target': torch.zeros(batch_size)
                }

        # On the controller
        with mesh.activate():
            dataloader = dataloader_pipe(1, 1)
            input, target = dataloader.recv()
    """

    def __init__(self, fn: Remote, max_messages: int, args, kwargs):
        mesh = device_mesh._active
        if mesh is None:
            raise ValueError(
                "Remote generators require an active device mesh (use `with mesh.activate():`"
            )
        mesh.define_remotely()

        def no_references(x):
            if isinstance(x, Referenceable):
                raise ValueError("Cannot pass references to external generators")

        flatten((args, kwargs), no_references)
        self._fake_pipe = FakePipe()
        if not isinstance(fn, Remote):
            raise TypeError("expected fn to be a monarch.remote function.")
        args_ = (self._fake_pipe, *args)
        # we do not pass references to generators so fake_args == args
        self._iterator = iter(fn._pipe_propagate(args_, kwargs, args_, kwargs))
        self.ref = mesh.client.new_ref()
        self.mesh = mesh
        key = f"ipc:///tmp/proc-{uuid.uuid4()}"
        self.mesh._send(
            messages.CreatePipe(
                self, key, fn._resolvable, max_messages, mesh, args, kwargs
            )
        )

    def send(self, obj: Any):
        client = self.mesh.client
        _fake_result, dtensors, _mutates, device_mesh = dtensor_check(
            (lambda args, kwargs, fake_args, fake_kwargs: fake_args[0]),
            ResolvableFunctionFromPath("ident"),
            (obj,),
            {},
            self.mesh,
            stream._active,
        )
        if self.mesh is not device_mesh:
            raise ValueError(
                f"Pipe is defined on mesh {self.mesh} but inputs are defined on mesh {device_mesh}"
            )
        self._fake_pipe._fake_sends.append(_fake_result)
        seq = client.new_node((), dtensors)
        self.mesh._send(
            messages.SendValue(
                seq, self, (), None, (obj,), {}, stream._active._to_ref(client)
            )
        )

    def recv(self) -> Any:
        mesh = self.mesh
        fake_result = fake_call(next, self._iterator)
        fake_result_tensors, unflatten = flatten(
            fake_result, lambda x: isinstance(x, torch.Tensor)
        )
        tensors = tuple(
            Tensor(fake, mesh, stream._active) for fake in fake_result_tensors
        )
        seq = mesh.client.new_node(tensors, ())
        result = unflatten(tensors)
        mesh._send(
            messages.PipeRecv(seq, result, self, stream._active._to_ref(mesh.client))
        )
        return result

    def delete_ref(self, ref: int):
        if not self.mesh.client._shutdown:
            self.mesh.client.handle_deletes(self.mesh.processes, [ref])

    # make typechecking happy for actual process functions
    @property
    def ranks(self) -> Dict["str", int]:
        raise ValueError("cannot be accessed on controller")

    @property
    def sizes(self) -> Dict["str", int]:
        raise ValueError("cannot be accessed on controller")


class FakePipe(Pipe):
    """
    Container to observe faked objects that the controller sent to the process
    """

    def __init__(self):
        self._fake_sends = deque[Any]()
        self.ref = None

    def send(self, obj: Any):
        raise RuntimeError(
            "Rather than p.send(x) use yield x to simulate a pipe worker sending data."
        )

    def recv(self):
        if self._fake_sends:
            return self._fake_sends.popleft()
