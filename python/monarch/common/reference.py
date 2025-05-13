# pyre-unsafe
from typing import Optional

from monarch._monarch.worker import Ref


class Referenceable:
    def __init__(self):
        self.ref: Optional[int] = None

    def delete_ref(self, ref):
        raise NotImplementedError("no delete_ref method")

    def __reduce_ex__(self, protocol):
        assert (
            self.ref is not None
        ), f"{self} is being sent but does not have a reference"
        return Ref, (self.ref,)

    # Used by rust backend to get the ref for this object
    def __monarch_ref__(self) -> int:
        assert self.ref is not None
        return self.ref

    def __del__(self):
        if self.ref is not None:
            self.delete_ref(self.ref)
