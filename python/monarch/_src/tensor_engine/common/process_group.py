# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging

import torch.distributed as dist

logger = logging.getLogger(__name__)


def _wrap_method(process_group: dist.ProcessGroup, method):
    def wrapper(*args, **kwargs):
        logger.debug(
            "ProcessGroup Call: %s with args %s and kwargs %s", method, args, kwargs
        )
        fn = getattr(process_group, method)
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.warning(
                "ProcessGroup Call: %s with args %s and kwargs %s failed with exception: %s",
                method,
                args,
                kwargs,
                str(e),
            )
            # TODO(rajeshn): send a message back to the controller that this
            # worker had a failed communication event
            raise e

    return wrapper


class SingleControllerProcessGroupWrapper:
    """
    Wraps a ProcessGroup object to provide a single controller process group. This provides us a hook to observe
    all the operatons on the process group to the controller.
    """

    def __new__(cls, pg: dist.ProcessGroup):
        instance = super().__new__(cls)

        for attr in dir(type(pg)):
            if not attr.startswith("__") and callable(getattr(type(pg), attr)):
                setattr(instance, attr, _wrap_method(pg, attr))

        return instance

    def __init__(self, process_group):
        self.process_group = process_group
