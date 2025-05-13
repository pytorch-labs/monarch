# pyre-unsafe
from typing import NamedTuple

from monarch_supervisor import get_message_queue


class Reply(NamedTuple):
    a: int
    b: int
    x: int


def reply_hello(a, b, x):
    q = get_message_queue()
    q.send(Reply(a, b, x))


def echo():
    q = get_message_queue()
    i = 0
    while True:
        sender, m = q.recv()
        if m == "exit":
            break
        assert m == i
        q.send(m)
        i += 1


class Mapper:
    def map(self, items):
        return sum(x * 2 for x in items)

    def reduce(self, items):
        return sum(items)

    def finish(self, result):
        return result
