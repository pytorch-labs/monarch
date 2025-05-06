import asyncio

from morpho.subprocess import call


def example_call(a, b):
    return a + b


def test_call():
    assert 7 == asyncio.run(call(example_call, 3, 4))
