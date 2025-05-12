# pyre-strict

from typing import cast, Literal
from unittest import TestCase

import pytest

from paft import simple_ftar


@pytest.mark.timeout(200)
class TestPaft(TestCase):
    # CTran is fairly unstable. It gets into all kinds of connection errors, NCCL timeout errors, etc.
    # Have retry here for mitigation.
    def test_paft_monarch_rust(self) -> None:
        max_retries = 3
        attempt = 0
        while attempt < max_retries:
            try:
                _run_paft_monarch("rust_local")
                break
            except Exception as e:
                attempt += 1
                if attempt == max_retries:
                    raise e  # Re-raise the exception if max retries are reached
                print(f"Retrying... Attempt {attempt} of {max_retries}")

    def test_paft_monarch_sim(self) -> None:
        _run_paft_monarch("sim")


def _run_paft_monarch(mesh_type: str) -> None:
    mesh_allreduce_map = {0: 2.0, 1: 3.0}
    simple_ftar.main(
        mesh_type=cast(Literal["python", "rust_local", "sim"], mesh_type),
        num_meshes=len(mesh_allreduce_map),
        mesh_to_allreduce_val=mesh_allreduce_map,
    )
