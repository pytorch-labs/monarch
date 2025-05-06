import io
import pickle
from typing import Any, Callable, Iterable, List, Tuple

import cloudpickle


class _Pickler(cloudpickle.Pickler):
    def __init__(self, filter):
        self.f = io.BytesIO()
        super().__init__(self.f)
        self._filter = filter
        self._saved = []

    def persistent_id(self, obj):
        if not self._filter(obj):
            return None
        self._saved.append(obj)
        return len(self._saved) - 1


class _Unpickler(pickle.Unpickler):
    def __init__(self, data, sequence: Iterable[Any]):
        super().__init__(io.BytesIO(data))
        self._iter = iter(sequence)
        self._values = []

    def persistent_load(self, id):
        while id >= len(self._values):
            self._values.append(next(self._iter))
        return self._values[id]


def flatten(obj: Any, filter: Callable[[Any], bool]) -> Tuple[List[Any], bytes]:
    pickler = _Pickler(filter)
    pickler.dump(obj)
    return pickler._saved, pickler.f.getvalue()


def unflatten(data: bytes, values: Iterable[Any]) -> Any:
    up = _Unpickler(data, values)
    return up.load()
