# pyre-unsafe

"""Tests monarch's rust CLI ('//monarch/tools:cli') by running it as a CLI.

The reason why the test for the rust CLI is in python is because we use cli_fallback
(https://fburl.com/wiki/gbgwgkmg) to run certain subcommands in Python as the libraries
these subcommands use only have a Python API and there's no good way to call Python
from Rust today in fbcode due to the way PARs work.

cli_fallback works by annotating the rust main function with a procedural macro
so there really isn't a good way to provide `std::env::args` programmatically
(at least not to also test fallback behavior). So we emulate a real CLI call
from python by using subprocesses.
"""

import importlib.resources
import json

import re
import subprocess
import unittest

_DISABLE_WORKSPACE = ""  # passing --workspace="" disables it for tests


def run(*cmd: str) -> str:
    return subprocess.check_output([*cmd], encoding="utf-8")


def rust_cli(*args) -> str:
    """Runs the Rust CLI with the provided arguments."""
    with importlib.resources.path("monarch.tools", "monarch") as rust_cli:
        return run(str(rust_cli), *args)


def py_cli(*args) -> str:
    """Runs the Python CLI with the provided arguments."""
    with importlib.resources.path("monarch.tools", "py_monarch") as py_cli:
        return run(str(py_cli), *args)


def normalize_json(_json, normalized_values=None):
    # looks for fields that are randomly generated
    # which are usually the fields with the name "*_id", "*Id"
    #
    # IMPORTANT: this function does NOT comprehensively normalize
    #  ALL randomly generated fields and might need adjustment
    #  if the scheduler request changes.

    if normalized_values is None:
        normalized_values = {}  # original_value -> normalized_value

    if isinstance(_json, list):
        for elem in _json:
            if isinstance(elem, (list | dict)):
                normalize_json(elem, normalized_values)
    elif isinstance(_json, dict):
        for key, value in _json.items():
            if isinstance(value, (list | dict)):
                normalize_json(value, normalized_values)
            else:
                if re.match(r".*_?(id|Id|ID)$", key):
                    if isinstance(value, str):
                        normalized_value = re.sub(r"[a-zA-Z0-9]", "X", value)
                        _json[key] = normalized_value
                        normalized_values[value] = normalized_value
                else:
                    if isinstance(value, str):
                        # value could have used macros (e.g. "/mnt/aidev/${job_name}")
                        normalized_value = value
                        for k, v in normalized_values.items():
                            if k in value:
                                normalized_value = normalized_value.replace(k, v)
                        _json[key] = normalized_value


class RustMainTest(unittest.TestCase):
    def test_py_cli_fallback(self) -> None:
        # checking a single fallback command suffices for coverage

        # useStrictName prevents suffixing job name with short uuid
        # for easier comparison
        args = (
            "create",
            "--dryrun",
            f"--workspace={_DISABLE_WORKSPACE}",
            "-cfg=useStrictName=True",
        )

        rust_out = json.loads(rust_cli(*args))
        py_out = json.loads(py_cli(*args))

        # normalizes randomly generated fields with `XXXX-XXXX`'s
        normalize_json(rust_out)
        normalize_json(py_out)

        self.assertEqual(rust_out, py_out)

    def test_rust_cli_bounce_should_fail(self) -> None:
        # TODO implement
        with self.assertRaises(subprocess.CalledProcessError):
            rust_cli("bounce", "mast_conda:///monarch-foo")

    def test_rust_cli_stop_should_fail(self) -> None:
        # TODO implement
        with self.assertRaises(subprocess.CalledProcessError):
            rust_cli("stop", "mast_conda:///monarch-foo")
