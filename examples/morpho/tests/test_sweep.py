import asyncio
import tempfile

from morpho.debug.sweep import local_debug_sweep


def test_sweep():
    with tempfile.NamedTemporaryFile("rb", suffix=".png", delete=False) as f:
        asyncio.run(local_debug_sweep(f.name))
        assert len(f.read()) != 0
        print(f.name)
