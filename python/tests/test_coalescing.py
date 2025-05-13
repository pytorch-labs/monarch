# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import itertools
from contextlib import contextmanager
from typing import List
from unittest import main, TestCase
from unittest.mock import patch

import monarch

import torch
from monarch import (
    coalescing,
    fetch_shard,
    get_active_mesh,
    get_active_stream,
    no_mesh,
    remote,
    Stream,
)
from monarch._testing import TestingContext
from monarch.common._coalescing import _record_and_define, compile
from monarch.common.function_caching import AliasOf, Storage, TensorGroup
from monarch.common.tensor import Tensor


def _do_bogus_tensor_work(x, y, fail_rank=None):
    return x + y  # real function actually does x @ y


do_bogus_tensor_work = remote(
    "monarch.worker._testing_function.do_bogus_tensor_work",
    propagate=_do_bogus_tensor_work,
)

log = remote("monarch.worker.worker.log", propagate="inspect")


def inspect(x):
    return fetch_shard(x).result().item()


def setUpModule():
    global local
    local = TestingContext().__enter__()


def tearDownModule():
    global local
    local.__exit__()


class TestCoalescing(TestCase):
    def setUp(self) -> None:
        print(f"Running test: {self._testMethodName}")
        return super().setUp()

    @classmethod
    def local_device_mesh(cls, N, gpu_per_host, activate=True):
        return local.local_device_mesh(N, gpu_per_host, activate)

    @property
    def num_outstanding_messages(self) -> int:
        return sum(
            len(msgs)
            for msgs in get_active_mesh().client.recorder.flat_messages.values()
        )

    def test_basic_coalescing(self) -> None:
        with self.local_device_mesh(1, 1):
            with coalescing():
                a = torch.zeros(3, 4)
                for _ in range(1, 10):
                    a = a + torch.ones(3, 4)
                # no messages should have been sient since coalescing is enabled
                self.assertGreaterEqual(self.num_outstanding_messages, 10)
            # now that the coalesce is done we should have flushed the messages
            self.assertEqual(self.num_outstanding_messages, 0)

    def test_repeat_simple(self) -> None:
        with self.local_device_mesh(1, 1):
            a = torch.zeros(())

            @compile(verify=False)
            def fn():
                nonlocal a
                z = torch.ones(())
                a += z
                return z

            z = None
            for _ in range(3):
                z = fn()

            self.assertEqual(inspect(a), 3)
            self.assertEqual(inspect(z), 1)

    def test_repeat_formals(self) -> None:
        with self.local_device_mesh(1, 1):
            a = torch.rand(3, 4)

            @compile(verify=False)
            def fn(a, b):
                return 2 * a + b

            for _ in range(3):
                b = torch.rand(3, 4)
                z = fn(a, b)
                lz, la, lb = monarch.inspect((z, a, b))
                assert isinstance(la, torch.Tensor)
                assert isinstance(lb, torch.Tensor)
                with no_mesh.activate():
                    self.assertTrue(torch.allclose(lz, 2 * la + lb))

            @compile(verify=False)
            def fn(b):
                return 2 * a + b

            for _ in range(3):
                b = torch.rand(3, 4)
                z = fn(b)
                lz, la, lb = monarch.inspect((z, a, b))
                assert isinstance(la, torch.Tensor)
                assert isinstance(lb, torch.Tensor)
                with no_mesh.activate():
                    self.assertTrue(torch.allclose(lz, 2 * la + lb))

    def test_repeat_error_inside(self) -> None:
        with self.local_device_mesh(1, 1):
            a = torch.zeros(())

            @compile(verify=False)
            def fn():
                nonlocal a
                z = torch.ones(())
                a += z
                do_bogus_tensor_work(z, z)
                return z

            z = fn()
            # recorded coalescing will lump errors together so check that
            with self.assertRaisesRegex(Exception, "both arguments to matmul"):
                inspect(z)

    def test_repeat_inner_borrow(self) -> None:
        with self.local_device_mesh(1, 1):
            a = torch.zeros(())
            other = Stream("other")
            with other.activate():
                b = torch.ones(())

            @compile(verify=False)
            def fn():
                nonlocal a, b
                c, borrow = get_active_stream().borrow(b)
                with borrow:
                    a += c

            for _ in range(3):
                fn()

            self.assertEqual(inspect(a), 3)

    def test_repeat_outer_borrow(self) -> None:
        with self.local_device_mesh(1, 1):
            a = torch.zeros(())
            other = Stream("other")
            with other.activate():
                b = torch.ones(())
            c, borrow = get_active_stream().borrow(b)

            @compile(verify=False)
            def fn():
                nonlocal a, c
                a += c
                z = torch.rand(3, 4)
                del c
                return z

            with borrow:
                z = None
                for _ in range(3):
                    z = fn()

            result = fetch_shard(a).result()
            fetch_shard(z).result()
            with no_mesh.activate():
                self.assertEqual(result.item(), 3)

    def test_nested_coalescing(self) -> None:
        with self.local_device_mesh(1, 1):
            with coalescing():
                a = torch.zeros(3, 4)
                with coalescing():
                    for _ in range(1, 10):
                        a = a + torch.ones(3, 4)
                    # confirm that there are messages awaiting to be send
                    self.assertGreaterEqual(
                        self.num_outstanding_messages,
                        10,
                    )
                # since we are in the nested block we shouldn't have flushed the messages yet
                self.assertGreaterEqual(self.num_outstanding_messages, 10)
            # now that the outer coalesce is done we should have flushed the messages
            self.assertEqual(self.num_outstanding_messages, 0)

    def test_no_coalescing(self) -> None:
        with self.local_device_mesh(1, 1):
            a = torch.zeros(3, 4)
            for _ in range(1, 10):
                a = a + torch.ones(3, 4)
            # without coalescing the messages should be sent with nothing outstanding
            self.assertEqual(self.num_outstanding_messages, 0)

    @contextmanager
    def assertRecorded(self, times: int):
        with patch(
            "monarch.common._coalescing._record_and_define",
            side_effect=_record_and_define,
        ) as m:
            yield
            self.assertEqual(m.call_count, times)

    def assertAliases(self, tensors: List[Tensor], aliasing: List[int]):
        group = TensorGroup([t._fake for t in tensors])
        c = iter(itertools.count())
        actual = []
        assert len(group.pattern.entries) == len(tensors)
        assert len(aliasing) == len(tensors)
        for e in group.pattern.entries:
            match e.storage:
                case AliasOf(offset=offset):
                    actual.append(offset)
                case Storage():
                    actual.append(next(c))
        self.assertEqual(aliasing, actual)

    def test_compile_aliasing(self) -> None:
        with self.local_device_mesh(1, 1):

            @compile(verify=False)
            def add(a, b):
                return a + b

            @compile(verify=False)
            def return_cond(a, b, c):
                if c:
                    return a
                else:
                    return b

            a = torch.rand(3, 4)
            b = torch.rand(3, 4)
            with self.assertRecorded(1):
                r = add(a, b)
                self.assertEqual(r.size(), (3, 4))
                r2 = add(b, a)
                self.assertAliases([a, b, r2, r], [0, 1, 2, 3])

            c = torch.rand(4)
            d = torch.rand(4, 4)
            with self.assertRecorded(1):
                e = add(c, d)
                self.assertEqual(e.size(), (4, 4))
                e = add(c, torch.rand(4, 4))
                self.assertEqual(e.size(), (4, 4))

            with self.assertRecorded(1):
                r = add(a, 4)
                self.assertAliases([r, a], [0, 1])

            with self.assertRecorded(1):
                r0 = return_cond(a, b, True)
                self.assertAliases([a, b, r0], [0, 1, 0])
                r1 = return_cond(b, a, True)
                self.assertAliases([a, b, r1], [0, 1, 1])

            with self.assertRecorded(1):
                r0 = return_cond(a, b, False)
                self.assertAliases([a, b, r0], [0, 1, 1])
                r1 = return_cond(a, b, False)
                self.assertAliases([b, a, r1], [0, 1, 0])

            @compile(verify=False)
            def captured(b):
                return a + b

            with self.assertRecorded(1):
                r = captured(b)
                self.assertAliases([a, b, r], [0, 1, 2])
                r = captured(torch.rand(3, 4))
                self.assertEqual(r.size(), (3, 4))

            with self.assertRecorded(1):
                # input aliased with capture
                captured(a)
                captured(a)

            @compile(verify=False)
            def weird(f, g):
                o = f + g
                return o, o[0], f[0], g[0], a[0]

            with self.assertRecorded(1):
                r0, r1, r2, r3, r4 = weird(c, d)
                self.assertAliases(
                    [c, d, a, r0, r1, r2, r3, r4], [0, 1, 2, 3, 3, 0, 1, 2]
                )

    def test_compile_input_permissions(self):
        with self.local_device_mesh(1, 1):
            a = torch.rand(3, 4)

            @compile(verify=False)
            def add(b):
                return a + b

            with self.assertRecorded(1):
                c = add(torch.rand(3, 4))

            other = Stream("other")
            ab, borrow = other.borrow(a, mutable=True)

            with borrow:
                with self.assertRaisesRegex(TypeError, "BORROWED"):
                    add(torch.rand(3, 4))

            # test we can read it again
            add(torch.rand(3, 4))

            ab, borrow = other.borrow(a)
            with borrow:
                add(torch.rand(3, 4))

            with self.assertRecorded(0):
                with other.activate():
                    c = torch.rand(3, 4)
                c, borrow = monarch.get_active_stream().borrow(c)
                with borrow:
                    add(c)

            a.drop()

            with self.assertRaisesRegex(TypeError, "DROPPED"):
                add(torch.rand(3, 4))

    def test_compile_verify(self):
        with self.local_device_mesh(1, 1):
            a = torch.rand(3, 4)

            @compile(verify=True)
            def add(b):
                return a + b

            c = False

            @compile(verify=True)
            def add_broken(b):
                nonlocal c
                if c:
                    a = torch.zeros(3, 4)
                else:
                    a = torch.rand(3, 4)
                return a.add(b)

            with self.assertRecorded(2):
                add(torch.rand(3, 4))
                add(torch.rand(3, 4))
                add(torch.rand(3, 4))

            add_broken(torch.rand(3, 4))
            with self.assertRaisesRegex(RuntimeError, "diverges"):
                c = True
                add_broken(torch.rand(3, 4))

    def test_dropped(self):
        with self.local_device_mesh(1, 1):
            a = torch.rand(3, 4)
            b = None

            @compile(verify=False)
            def foo():
                nonlocal b
                b = a + a

            foo()
            with self.assertRaisesRegex(TypeError, "DROPPED"):
                b.add(4)

    def test_across_mesh(self):
        with self.local_device_mesh(1, 2) as m:
            m0 = m(gpu=0)
            m1 = m(gpu=1)

            @compile
            def foo(a, b):
                with m0.activate():
                    r0 = a + a
                with m1.activate():
                    r1 = b + b
                return r0, r1

            with m0.activate():
                a = torch.rand(3, 4)
            with m1.activate():
                b = torch.rand(3, 4)

            r0, r1 = foo(a, b)
            with m0.activate():
                monarch.inspect(r0)
            with m1.activate():
                monarch.inspect(r0)

    def test_grad_not_supported(self):
        with self.local_device_mesh(1, 1):

            @compile
            def foo(x):
                return x

            y = torch.rand(3, requires_grad=True)

            @compile
            def returnit():
                return y

            with self.assertRaisesRegex(TypeError, "REQUIRES_GRAD"):
                foo(torch.rand(3, requires_grad=True))

            with self.assertRaisesRegex(TypeError, "REQUIRES_GRAD"):
                returnit()

    def test_mutate_inputs(self):
        with self.local_device_mesh(1, 1) as mesh:

            @compile(verify=False)
            def foo(x_not_mutated, w_not_mutated, y, y_alias, z, z_alias):
                u = (
                    x_not_mutated.mul(2.0)
                    + w_not_mutated
                    + z_alias.unsqueeze(0).repeat(3, 1)
                )
                v = y.add(5.0)
                stream = monarch.Stream("borrow")
                borrowed_y_alias, y_alias_borrow = stream.borrow(y_alias, mutable=True)
                with stream.activate():
                    borrowed_y_alias.add_(1.0)
                y_alias_borrow.drop()
                z.add_(1.0)
                return u, v

            x_not_mutated = torch.rand(3, 3)
            w_not_mutated = torch.rand(3, 3)
            y = torch.rand(3, 3)
            y_alias = y.reshape(-1)
            z = torch.rand(3, 3)
            z_alias = z[0, :]

            mutated_inputs = (y, y_alias, z, z_alias)
            mutated_aliases = set().union(*[t._aliases.aliases for t in mutated_inputs])
            all_inputs = (x_not_mutated, w_not_mutated) + mutated_inputs
            with patch.object(
                mesh.client,
                "new_node_nocoalesce",
                side_effect=mesh.client.new_node_nocoalesce,
            ) as new_node:
                for _ in range(2):
                    u, v = foo(*all_inputs)
                    (mutated, used, _, _), _ = new_node.call_args
                    self.assertEqual(
                        mutated_aliases.union(u._aliases.aliases, v._aliases.aliases),
                        set(mutated),
                    )
                    self.assertEqual(set(all_inputs), set(used))


if __name__ == "__main__":
    main()
