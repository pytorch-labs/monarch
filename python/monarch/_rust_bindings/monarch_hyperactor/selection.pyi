# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

@final
class Selection:
    """Opaque representation of a selection expression used to represent
    constraints over multidimensional shapes.

    Construct via from_string()` and use with mesh APIs to filter,
    evaluate, or route over structured topologies.
    """
    def __repr__(self) -> str: ...
    @classmethod
    def from_string(cls, s: str) -> Selection:
        """Parse a selection expression from a string.

        Accepts a compact string syntax such as `"(*, 0:4)"` or `"0 & (1 | 2)"`,
            and returns a structured Selection object.

        Raises:
            ValueError: if the input string is not a valid selection expression.
        """
        ...
