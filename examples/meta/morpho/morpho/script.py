# pyre-unsafe
import importlib
import sys
from pathlib import Path

from morpho.cli import typer_app


def help():
    print("""\
usage: morpho path.to.module.function

example: morpho morpho.triton.train --help
""")


def main():
    sys.path.append(".")
    import os

    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    if len(sys.argv) < 2 or "." not in sys.argv[1]:
        help()
        sys.exit(1)
    function = sys.argv[1]
    module, function = function.rsplit(".", maxsplit=1)
    impl = getattr(importlib.import_module(module), function)
    exename = str(Path(sys.argv[0]).name)
    prog_name = f"{exename} {function}"
    typer_app(impl)(sys.argv[2:], prog_name=prog_name)
