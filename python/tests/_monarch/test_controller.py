# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from unittest import TestCase

from monarch._monarch import controller, shape, worker


class TestController(TestCase):
    def test_node(self) -> None:
        node = controller.Node(seq=10, defs=[worker.Ref(id=1)], uses=[])
        self.assertEqual(node.seq, 10)
        self.assertEqual(node.defs, [worker.Ref(id=1)])
        self.assertEqual(node.uses, [])

    def test_send(self) -> None:
        msg = controller.Send(
            ranks=[shape.Slice(offset=1, sizes=[3], strides=[1])],
            message=worker.CreateStream(
                id=worker.StreamRef(id=10),
                stream_creation=worker.StreamCreationMode.CreateNewStream,
            ),
        )
        self.assertEqual(len(msg.ranks), 1)
        self.assertEqual(msg.ranks[0].ndim, 1)
        message = msg.message
        assert isinstance(message, worker.CreateStream)
        self.assertEqual(
            message.stream_creation, worker.StreamCreationMode.CreateNewStream
        )
        self.assertEqual(message.id, worker.StreamRef(id=10))

    def test_send_no_ranks(self) -> None:
        with self.assertRaises(ValueError):
            controller.Send(
                ranks=[],
                message=worker.CreateStream(
                    id=worker.StreamRef(id=10),
                    stream_creation=worker.StreamCreationMode.CreateNewStream,
                ),
            )

    def test_node_serde(self) -> None:
        node = controller.Node(seq=10, defs=[worker.Ref(id=1)], uses=[])
        serialized = node.serialize()
        deserialized = controller.Node.from_serialized(serialized)
        self.assertEqual(node.seq, deserialized.seq)
        self.assertEqual(node.defs, deserialized.defs)
        self.assertEqual(node.uses, deserialized.uses)

    def test_send_serde(self) -> None:
        msg = controller.Send(
            ranks=[shape.Slice(offset=1, sizes=[3], strides=[1])],
            message=worker.CreateStream(
                id=worker.StreamRef(id=10),
                stream_creation=worker.StreamCreationMode.CreateNewStream,
            ),
        )
        serialized = msg.serialize()
        deserialized = controller.Send.from_serialized(serialized)
        self.assertEqual(len(msg.ranks), len(deserialized.ranks))
        self.assertEqual(msg.ranks[0].ndim, deserialized.ranks[0].ndim)
        message = msg.message
        deser_message = deserialized.message
        assert isinstance(message, worker.CreateStream)
        assert isinstance(deser_message, worker.CreateStream)
        self.assertEqual(message.stream_creation, deser_message.stream_creation)
        self.assertEqual(message.id, deser_message.id)
