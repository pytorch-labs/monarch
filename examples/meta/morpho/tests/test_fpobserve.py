import tempfile

import torch
from morpho.debug.observer import Observer, plot_runs


def test_fpobserver():
    runs = []

    for name in ["test1", "test2"]:
        for i in range(0, 3):
            for v in ["reference", "reassociated", "noise"]:
                if v == "reference" and i > 0:
                    continue
                o = Observer(name, v, i)
                for j in range(10):
                    o.observe("activation", torch.rand(256, 256, device="cuda"))
                runs.append(o.run)
    with tempfile.NamedTemporaryFile("rb", delete=False, suffix=".png") as f:
        plot_runs(runs, f.name)
        assert len(f.read()) != 0
