# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""
This is a utility file for fetching a shard of a tensor from remote.
"""

from typing import cast, TypeVar

from monarch.common.device_mesh import no_mesh

from monarch.common.future import Future

from monarch.common.remote import call_on_shard_and_fetch, remote_identity

T = TypeVar("T")


def fetch_shard(
    obj: T, shard: dict[str, int] | None = None, **kwargs: int
) -> Future[T]:
    """
    Retrieve the shard at `coordinates` of the current device mesh of each
    tensor in obj. All tensors in `obj` will be fetched to the CPU device.
        obj - a pytree containing the tensors the fetch
        shard - a dictionary from mesh dimension name to coordinate of the shard
                If None, this will fetch from coordinate 0 for all dimensions (useful after all_reduce/all_gather)
        preprocess - a
        **kwargs - additional keyword arguments are added as entries to the shard dictionary
    """
    if kwargs:
        if shard is None:
            shard = {}
        shard.update(kwargs)

    return cast("Future[T]", call_on_shard_and_fetch(remote_identity, obj, shard=shard))


def show(obj: T, shard: dict[str, int] | None = None, **kwargs: int) -> object:
    """
    Fetch and visualize a shard of tensors using torchshow.

    This function fetches the specified shard from remote processes and displays it
    using the torchshow library for visualization.

    Args:
        obj: A pytree containing tensors to fetch and display.
        shard: Dictionary mapping mesh dimension names to coordinates.
               If None, fetches from coordinate 0 for all dimensions.
        **kwargs: Additional shard coordinates as keyword arguments.

    Returns:
        The visualization result from torchshow.

    Example:
        >>> # Show shard at coordinate (0, 1) for dimensions 'batch' and 'gpu'
        >>> show(tensor, shard={'batch': 0, 'gpu': 1})
        >>> # Equivalent using kwargs
        >>> show(tensor, batch=0, gpu=1)
    """
    v = inspect(obj, shard=shard, **kwargs)
    # pyre-ignore
    from torchshow import show  # @manual

    with no_mesh.activate():
        return show(v)


def inspect(obj: T, shard: dict[str, int] | None = None, **kwargs: int) -> T:
    """
    Fetch and return a shard of tensors from remote processes.

    This function synchronously fetches the specified shard from remote processes
    and returns the result. All tensors will be moved to the CPU device.

    Args:
        obj: A pytree containing tensors to fetch.
        shard: Dictionary mapping mesh dimension names to coordinates.
               If None, fetches from coordinate 0 for all dimensions.
        **kwargs: Additional shard coordinates as keyword arguments.

    Returns:
        The fetched shard with all tensors on CPU.

    Example:
        >>> # Fetch shard at coordinate (2, 0) for dimensions 'batch' and 'gpu'
        >>> result = inspect(tensor, shard={'batch': 2, 'gpu': 0})
        >>> # Equivalent using kwargs
        >>> result = inspect(tensor, batch=2, gpu=0)
    """
    return fetch_shard(obj, shard=shard, **kwargs).result()
