# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from functools import wraps


class _ContextManager:
    def __init__(self, generator):
        self.generator = generator
        self.generator.send(None)

    def __enter__(self):
        return

    def __exit__(self, *args):
        try:
            self.generator.send(None)
        except StopIteration:
            pass
        else:
            raise RuntimeError("context manager generator did not exit")


def activate_first_context_manager(func):
    """
    Similar to contextlib.contextmanager but it
    starts the context when the function is called rather than
    than at the start of the with statement. Useful for things where
    you want to optionally activate the context without a guard.
    """

    @wraps(func)
    def helper(*args, **kwargs):
        return _ContextManager(func(*args, **kwargs))

    return helper
