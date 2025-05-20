from typing import Dict, final, Optional

class Alloc:
    """
    An alloc represents an allocation of procs. Allocs are returned by
    one of the allocator implementations, such as `ProcessAllocator` or
    `LocalAllocator`.
    """

@final
class AllocConstraints:
    def __init__(self, match_labels: Optional[Dict[str, str]] = None) -> None:
        """
        Create a new alloc constraints.

        Arguments:
        - `match_labels`: A dictionary of labels to match. If a label is present
                in the dictionary, the alloc must have that label and its value
                must match the value in the dictionary.
        """
        ...

@final
class AllocSpec:
    def __init__(self, constraints: AllocConstraints, **kwargs: int) -> None:
        """
        Initialize a shape with the provided dimension-size pairs.
        For example, `AllocSpec(constraints, replica=2, host=3, gpu=8)` creates a
        shape with 2 replicas with 3 hosts each, each of which in turn
        has 8 GPUs.
        """
        ...
