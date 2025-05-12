# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import os
import subprocess
import time
from unittest import TestCase

from parameterized import parameterized


class TestControllerExample(TestCase):
    @parameterized.expand(
        [
            "PYTHON_LOCAL",
            "RUST_TEST",
        ]
    )
    def test_controller_example(self, mesh_type: str) -> None:
        os.environ["MESH_TYPE"] = mesh_type
        env_var = "EXAMPLE_BINARY"
        example_binary = os.environ.get(env_var)
        start = time.time()
        if example_binary:
            # Run the executable
            try:
                # Make sure the process can be executed and exit without an issue
                subprocess.run([example_binary], check=True)
            except subprocess.CalledProcessError as e:
                raise Exception(f"Command failed with return code {e.returncode}")
            except Exception as e:
                raise Exception(f"Error running command: {e}")
        else:
            raise Exception(f"Environment variable {env_var} not set")

        # For now, simply test we can complete the script in 150 seconds
        self.assertLess(time.time() - start, 150)
