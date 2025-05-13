import collections.abc

from .._lib import shape  # @manual=//monarch/monarch_extension:monarch_extension

Slice = shape.Slice
Shape = shape.Shape


class Point(shape.Point, collections.abc.Mapping):
    pass


__all__ = ["Slice", "Shape"]
