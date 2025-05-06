# pyre-unsafe
from typing import List, Protocol


class Droppable(Protocol):
    """
    A handle to some resource that needs to be explicitly dropped later.
    This can be done either by using it as context manager:
        with droppable:
            ...
    Or by explicitly calling its drop method later:
        droppable.drop()
    Droppable objects will issue warnings if they get deleted implicitly
    without the above actions happening.
    """

    def __enter__(self):
        pass

    def __exit__(self, exc, exc_value, traceback):
        self.drop()

    def drop(self):
        pass


class NullDroppable(Droppable):
    pass


class DroppableList(List[Droppable], Droppable):
    def drop(self):
        for item in self:
            item.drop()


null_droppable = NullDroppable()
