import collections.abc

from monarch._rust_bindings.monarch_hyperactor.shape import (  # @manual=//monarch/monarch_extension:monarch_extension
    Point as _Point,
    Shape,
    Slice,
)


class Point(_Point, collections.abc.Mapping):
    pass


__all__ = ["Slice", "Shape", "Point"]
