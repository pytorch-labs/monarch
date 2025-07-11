# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections.abc
from typing import Dict, final, Iterator, List, overload, Sequence

@final
class Slice:
    """
    A wrapper around [ndslice::Slice] to expose it to python.
    It is a compact representation of indices into the flat
    representation of an n-dimensional array. Given an offset, sizes of
    each dimension, and strides for each dimension, Slice can compute
    indices into the flat array.

    Arguments:
    - `offset`: Offset into the flat array.
    - `sizes`: Sizes of each dimension.
    - `strides`: Strides for each dimension.
    """

    def __init__(
        self, *, offset: int, sizes: list[int], strides: list[int]
    ) -> None: ...
    @property
    def ndim(self) -> int:
        """The number of dimensions in the slice."""
        ...

    @property
    def offset(self) -> int:
        """
        The offset of the slice i.e. the first number from which
        values in the slice begin.
        """
        ...

    @property
    def sizes(self) -> list[int]:
        """The sizes of each dimension in the slice."""
        ...

    @property
    def strides(self) -> list[int]:
        """The strides of each dimension in the slice."""
        ...

    def index(self, value: int) -> int:
        """
        Returns the index of the given `value` in the slice or raises a
        `ValueError` if `value` is not in the slice.
        """
        ...

    def coordinates(self, value: int) -> list[int]:
        """
        Returns the coordinates of the given `value` in the slice or raises a
        `ValueError` if the `value` is not in the slice.
        """
        ...

    def nditem(self, coordinates: list[int]) -> int:
        """
        Returns the value at the given `coordinates` or raises an `IndexError`
        if the `coordinates` are out of bounds.
        """
        ...

    def __eq__(self, value: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __getnewargs_ex__(self) -> tuple[tuple, dict]: ...
    @overload
    def __getitem__(self, i: int) -> int: ...
    @overload
    def __getitem__(self, i: slice) -> tuple[int, ...]: ...
    def __len__(self) -> int:
        """Returns the complete size of the slice."""
        ...

    def __iter__(self) -> Iterator[int]:
        """Returns an iterator over the values in the slice."""
        ...

    @staticmethod
    def from_list(ranks: list[int]) -> list["Slice"]:
        """Returns a list of slices that cover the given list of ranks."""
        ...

    @staticmethod
    def new_row_major(ranks: Sequence[int]) -> "Slice":
        """Returns a contiguous slice composed of ranks"""
        ...

    def __repr__(self) -> str: ...

@final
class Shape:
    """
    A datastructure to represent a named and ordered list of labels (dimensions)
    and the Shape these labels represent.

    Arguments:
    - `labels`: A list of strings representing the labels for each dimension.
    - `slice`: An Slice object representing the shape.
    """
    def __new__(cls, labels: Sequence[str], slice: Slice) -> "Shape": ...
    @property
    def ndslice(self) -> Slice: ...
    @property
    def labels(self) -> List[str]:
        """The labels for each dimension of ndslice (e.g. "host", "gpu")"""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def coordinates(self, rank: int) -> Dict[str, int]:
        """
        Get the coordinates (e.g. {gpu:0, host:3}) where rank `rank` occurs in this shape.
        """
        ...

    def index(self, **kwargs: Dict[str, int]) -> "Shape":
        """
        Create a sub-slice of this shape:
            new_shape = shape.index(gpu=3, host=0)
        `new_shape` will no longer have gpu or host dimensions.
        """
        ...
    @staticmethod
    def from_bytes(bytes: bytes) -> "Shape": ...
    def ranks(self) -> List[int]:
        """
        Create an explicit list of all the ranks included in this Shape
        """
        ...
    def __len__(self) -> int: ...
    @staticmethod
    def unity() -> "Shape": ...

class Point(collections.abc.Mapping):
    """
    A point inside a multidimensional shape. The rank is the offset into the
    ndslice that selects which member of the ndslice this point represents.
    Point behaves like a dictionary, taking named dimensions and returning
    the index into those dimensions.

    It also additionally lets someone get the underlying shape.
    """
    def __new__(cls, rank: int, shape: "Shape") -> "Point": ...
    def __getitem__(self, label: str) -> int: ...
    def __len__(self) -> int: ...
    def size(self, label: str) -> int: ...
    @property
    def rank(self) -> int: ...
    @property
    def shape(self) -> "Shape": ...
