# pyre-unsafe

from typing import Generator

from monarch._monarch.shape import Shape, Slice

NDSlice = Slice

Ranks = Slice | list[Slice]


def iter_ranks(ranks: Ranks) -> Generator[int, None, None]:
    if isinstance(ranks, list):
        seen = set()
        for slice_ in ranks:
            for rank in slice_:
                if rank not in seen:
                    seen.add(rank)
                    yield rank
    else:
        yield from ranks


__all__ = ["NDSlice", "Shape"]
