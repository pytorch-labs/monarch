# pyre-unsafe
import logging
import traceback
from collections import defaultdict
from typing import Dict, Generator, List, NamedTuple, Tuple, TYPE_CHECKING, Union

from monarch.common.ndslice import iter_ranks

from monarch.common.tensor import InputChecker

from . import messages

if TYPE_CHECKING:
    from monarch.common.client import Client

from .ndslice import NDSlice
from .reference import Referenceable
from .tensor import Tensor

logger = logging.getLogger(__name__)


def flatten_messages(
    messages: List[Tuple[Union[NDSlice, List[NDSlice]], NamedTuple]],
) -> Dict[int, List[NamedTuple]]:
    result: Dict[int, List[NamedTuple]] = defaultdict(list)
    for ranks, msg in messages:
        for rank in iter_ranks(ranks):
            result[rank].append(msg)
    return result


class Recording(Referenceable):
    def __init__(
        self,
        client: "Client",
        uses: List["Tensor"],
        mutates: List["Tensor"],
        mutated_formal_indices: List[int],
        tracebacks: List[List[traceback.FrameSummary]],
        buffered_messages: List[Tuple[Union[NDSlice, List[NDSlice]], NamedTuple]],
        nresults: int,
        nformals: int,
        first_ref: int,
    ):
        self.uses = uses
        self.mutates = mutates
        # on future invocations of this recording, new aliases for our mutated tensors exists
        # an we will technically mutate them as well. This would be simplified and faster if our
        # node tracking worked with storages rather than tensors, but for now we have to collect
        # all the aliases on each invocation
        self.mutate_aliases = [m._aliases.aliases for m in self.mutates]
        self.mutated_formal_indices = mutated_formal_indices
        self.tracebacks = tracebacks
        self.ref = client.new_ref()
        self.first_ref = first_ref
        self.client = client
        self.buffered_messages = buffered_messages
        flat_messages = flatten_messages(self.buffered_messages)
        self.ranks = NDSlice.from_list(sorted(flat_messages.keys()))
        for rank, msgs in flat_messages.items():
            self.client.send_nocoalesce(
                NDSlice(offset=rank, sizes=[], strides=[]),
                messages.DefineRecording(self, nresults, nformals, msgs),
            )

    def run(self, results: Generator[Tensor, None, None], actuals: List[Tensor]):
        all_uses: List[Tensor] = [*self.uses, *actuals]
        with InputChecker.from_flat_args(
            "recording", all_uses, lambda ts: (tuple(ts), {})
        ) as checker:
            mutates_actuals = [
                actuals[i]._aliases.aliases for i in self.mutated_formal_indices
            ]
            mutates = list(set().union(*self.mutate_aliases, *mutates_actuals))
            checker.check_permission(mutates)
        # we are careful to not generate the results tensors until
        # after the input checker so that we do not create tensor objects
        # for tensors that will never be defined by CallRecording
        results_tuple = list(results)
        seq = self.client.new_node(
            results_tuple + mutates,
            all_uses,
            None,
            self.tracebacks,
        )
        self.client.send(
            self.ranks, messages.CallRecording(seq, self, results_tuple, actuals)
        )
        return results_tuple

    def delete_ref(self, ref: int):
        if not self.client.has_shutdown:
            self.client.handle_deletes(self.ranks, [ref])
