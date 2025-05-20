# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import math
from typing import cast
from unittest import TestCase

import cloudpickle

from monarch._monarch import shape, worker
from pyre_extensions import none_throws


def is_nan(val: int) -> bool:
    return math.isnan(val)


class MockReferencable:
    def __init__(self, ref: int) -> None:
        self.ref = ref

    def __monarch_ref__(self) -> int:
        return self.ref


class TestWorker(TestCase):
    def test_backend_network_init(self) -> None:
        msg = worker.BackendNetworkInit()
        self.assertTrue(isinstance(msg, worker.WorkerMessage))

    def test_backend_network_init_point_to_point(self) -> None:
        msg = worker.BackendNetworkPointToPointInit(
            from_stream=worker.StreamRef(id=1), to_stream=worker.StreamRef(id=2)
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.from_stream.id, 1)
        self.assertEqual(msg.to_stream.id, 2)

    def test_call_function(self) -> None:
        import torch

        msg = worker.CallFunction(
            seq=10,
            results=[worker.Ref(id=2), None],
            mutates=[],
            function=worker.FunctionPath(path="torch.ops.aten.ones.default"),
            args=([2, 3],),
            kwargs={"device": "cpu", "pin_memory": False},
            stream=worker.StreamRef(id=1),
            remote_process_groups=[],
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.seq, 10)
        self.assertEqual(msg.results, [worker.Ref(id=2), None])
        self.assertEqual(msg.mutates, [])
        self.assertEqual(
            cast(worker.FunctionPath, msg.function).path, "torch.ops.aten.ones.default"
        )
        self.assertEqual(msg.args, ([2, 3],))
        self.assertEqual(
            msg.kwargs, {"device": torch.device("cpu"), "pin_memory": False}
        )
        self.assertIsInstance(msg.kwargs["pin_memory"], bool)
        # we cannot use isinstance to assert bool vs int
        self.assertTrue(msg.kwargs["pin_memory"] is False)
        self.assertEqual(msg.stream, worker.StreamRef(id=1))

    def test_call_function_live_function(self) -> None:
        msg = worker.CallFunction(
            seq=10,
            results=[],
            mutates=[],
            function=worker.Cloudpickle(bytes=cloudpickle.dumps(is_nan)),
            args=(),
            kwargs={},
            stream=worker.StreamRef(id=1),
            remote_process_groups=[],
        )
        self.assertFalse(msg.function.resolve()(4))

    def test_call_function_referencable_args(self) -> None:
        msg = worker.CallFunction(
            seq=10,
            results=[],
            mutates=[],
            function=worker.FunctionPath(path="torch.ops.aten._foreach_add.Tensor"),
            args=([MockReferencable(1), MockReferencable(2)],),
            kwargs={
                "other": MockReferencable(3),
            },
            stream=worker.StreamRef(id=1),
            remote_process_groups=[],
        )
        self.assertEqual(msg.args, ([worker.Ref(id=1), worker.Ref(id=2)],))
        self.assertEqual(msg.kwargs, {"other": worker.Ref(id=3)})

    def test_create_stream(self) -> None:
        msg = worker.CreateStream(
            id=worker.StreamRef(id=10),
            stream_creation=worker.StreamCreationMode.CreateNewStream,
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.id.id, 10)
        self.assertEqual(msg.stream_creation, worker.StreamCreationMode.CreateNewStream)

    def test_create_device_mesh(self) -> None:
        msg = worker.CreateDeviceMesh(
            result=worker.Ref(id=10),
            names=("x", "y"),
            ranks=shape.Slice(offset=0, sizes=[2, 3], strides=[3, 1]),
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.result.id, 10)
        self.assertEqual(msg.names, ["x", "y"])
        self.assertEqual(msg.ranks.ndim, 2)

    def test_create_remote_process_group(self) -> None:
        msg = worker.CreateRemoteProcessGroup(
            result=worker.Ref(id=10),
            device_mesh=worker.Ref(id=12),
            dims=("x", "y"),
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.result.id, 10)
        self.assertEqual(msg.device_mesh.id, 12)
        self.assertEqual(msg.dims, ["x", "y"])

    def test_borrow_create(self) -> None:
        msg = worker.BorrowCreate(
            result=worker.Ref(id=10),
            borrow=23,
            tensor=worker.Ref(id=20),
            from_stream=worker.StreamRef(id=1),
            to_stream=worker.StreamRef(id=2),
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.result.id, 10)
        self.assertEqual(msg.borrow, 23)
        self.assertEqual(msg.tensor.id, 20)
        self.assertEqual(msg.from_stream.id, 1)
        self.assertEqual(msg.to_stream.id, 2)

    def test_borrow_first_use(self) -> None:
        msg = worker.BorrowFirstUse(
            borrow=23,
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.borrow, 23)

    def test_borrow_last_use(self) -> None:
        msg = worker.BorrowLastUse(
            borrow=23,
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.borrow, 23)

    def test_borrow_drop(self) -> None:
        msg = worker.BorrowDrop(
            borrow=23,
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.borrow, 23)

    def test_delete_refs(self) -> None:
        msg = worker.DeleteRefs(refs=[worker.Ref(id=1), worker.Ref(id=2)])
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.refs, [worker.Ref(id=1), worker.Ref(id=2)])

    def test_request_status(self) -> None:
        msg = worker.RequestStatus(seq=10, controller=False)
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.seq, 10)

    def test_reduce(self) -> None:
        import torch

        msg = worker.Reduce(
            result=worker.Ref(id=10),
            tensor=worker.Ref(id=20),
            factory=worker.TensorFactory(
                size=(2, 3),
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                layout=torch.sparse_csr,
            ),
            mesh=worker.Ref(id=30),
            dims=("x",),
            stream=worker.StreamRef(id=1),
            scatter=False,
            in_place=False,
            reduction=worker.ReductionType.Stack,
            out=None,
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.result.id, 10)
        self.assertEqual(msg.tensor.id, 20)
        self.assertEqual(msg.factory.size, (2, 3))
        self.assertEqual(msg.factory.dtype, torch.bfloat16)
        self.assertEqual(msg.factory.device, str(torch.device("cpu")))
        self.assertEqual(msg.factory.layout, torch.sparse_csr)
        self.assertEqual(msg.mesh.id, 30)
        self.assertEqual(msg.dims, ["x"])
        self.assertEqual(msg.stream.id, 1)
        self.assertEqual(msg.scatter, False)
        self.assertEqual(msg.in_place, False)
        self.assertEqual(msg.reduction, worker.ReductionType.Stack)
        self.assertIsNone(msg.out)

        msg = worker.Reduce(
            result=worker.Ref(id=10),
            tensor=worker.Ref(id=20),
            factory=worker.TensorFactory(
                size=(2, 3),
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                layout=torch.sparse_csr,
            ),
            mesh=worker.Ref(id=30),
            dims=("x",),
            stream=worker.StreamRef(id=1),
            scatter=False,
            in_place=False,
            reduction=worker.ReductionType.Stack,
            out=worker.Ref(id=40),
        )
        self.assertEqual(msg.out.id, 40)

    def test_send_tensor(self) -> None:
        import torch

        msg = worker.SendTensor(
            tensor=worker.Ref(id=10),
            from_stream=worker.StreamRef(id=1),
            to_stream=worker.StreamRef(id=2),
            from_ranks=shape.Slice(offset=0, sizes=[2, 3], strides=[3, 1]),
            to_ranks=shape.Slice(offset=0, sizes=[3, 4, 5], strides=[20, 5, 1]),
            result=worker.Ref(id=2),
            factory=worker.TensorFactory(
                size=(2, 5),
                dtype=torch.float32,
                device=torch.device("cuda"),
                layout=torch.strided,
            ),
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.tensor.id, 10)
        self.assertEqual(msg.from_stream.id, 1)
        self.assertEqual(msg.to_stream.id, 2)
        self.assertEqual(msg.from_ranks.ndim, 2)
        self.assertEqual(msg.to_ranks.ndim, 3)
        self.assertEqual(msg.result.id, 2)
        self.assertEqual(msg.factory.size, (2, 5))
        self.assertEqual(msg.factory.dtype, torch.float32)

    def test_create_pipe(self) -> None:
        msg = worker.CreatePipe(
            result=worker.Ref(id=10),
            key="some_key",
            function=worker.FunctionPath(path="builtins.range"),
            max_messages=1,
            mesh=worker.Ref(id=20),
            args=(1, 10),
            kwargs={},
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.result.id, 10)
        self.assertEqual(msg.key, "some_key")
        self.assertEqual(cast(worker.FunctionPath, msg.function).path, "builtins.range")
        self.assertEqual(msg.mesh.id, 20)
        self.assertEqual(msg.args, (1, 10))
        self.assertEqual(msg.kwargs, {})

    def test_send_value(self) -> None:
        msg = worker.SendValue(
            seq=100,
            destination=worker.Ref(id=10),
            mutates=[],
            function=None,
            args=(500,),
            kwargs={},
            stream=worker.StreamRef(id=1),
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.seq, 100)
        self.assertEqual(none_throws(msg.destination).id, 10)
        self.assertEqual(msg.mutates, [])
        self.assertEqual(msg.function, None)
        self.assertEqual(msg.args, (500,))
        self.assertEqual(msg.kwargs, {})
        self.assertEqual(msg.stream.id, 1)

    def test_pipe_recv(self) -> None:
        msg = worker.PipeRecv(
            seq=101,
            results=[worker.Ref(id=10), worker.Ref(id=11)],
            pipe=worker.Ref(id=1),
            stream=worker.StreamRef(id=2),
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(msg.seq, 101)
        self.assertEqual(msg.results, [worker.Ref(id=10), worker.Ref(id=11)])
        self.assertEqual(msg.pipe.id, 1)
        self.assertEqual(msg.stream.id, 2)

    def test_exit(self) -> None:
        msg = worker.Exit(error_reason=None)
        self.assertTrue(isinstance(msg, worker.WorkerMessage))

    def test_command_group(self) -> None:
        msg = worker.CommandGroup(
            commands=[
                worker.CallFunction(
                    seq=10,
                    results=[worker.Ref(id=2), None],
                    mutates=[],
                    function=worker.FunctionPath(path="torch.ops.aten.ones.default"),
                    args=([2, 3],),
                    kwargs={"device": "cpu"},
                    stream=worker.StreamRef(id=1),
                    remote_process_groups=[],
                ),
                worker.Exit(error_reason=None),
            ]
        )
        self.assertTrue(isinstance(msg, worker.WorkerMessage))
        self.assertEqual(len(msg.commands), 2)
        msg0 = msg.commands[0]
        assert isinstance(msg0, worker.CallFunction)
        self.assertEqual(msg0.seq, 10)
        self.assertEqual(msg0.results, [worker.Ref(id=2), None])
        self.assertEqual(
            cast(worker.FunctionPath, msg0.function).path, "torch.ops.aten.ones.default"
        )
        self.assertEqual(msg0.args, ([2, 3],))
        self.assertTrue(isinstance(msg.commands[1], worker.Exit))
