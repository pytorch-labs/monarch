# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import os
import pathlib
from typing import Optional


def MONARCH_HOME(*subdir_paths: str) -> pathlib.Path:
    """
    Path to the "dot-directory" for monarch.
    Defaults to `~/.monarch` and is overridable via the `MONARCH_HOME` environment variable.

    Usage:

    .. doc-test::

        from pathlib import Path
        from monarch.tools.utils import MONARCH_HOME

        assert MONARCH_HOME() == Path.home() / ".monarch"
        assert MONARCH_HOME("conda-pack-out") ==  Path.home() / ".monarch" / "conda-pack-out"
    ```
    """

    default_dir = str(pathlib.Path.home() / ".monarch")
    monarch_home = pathlib.Path(os.getenv("MONARCH_HOME", default_dir))

    monarch_home_subdir = monarch_home / os.path.sep.join(subdir_paths)
    monarch_home_subdir.mkdir(parents=True, exist_ok=True)

    return monarch_home_subdir


class conda:
    """Conda related util functions."""

    @staticmethod
    def active_env_dir() -> Optional[str]:
        """
        Returns the currently active conda environment's directory.
        `None` if run outside of a conda environment.
        """
        return os.getenv("CONDA_PREFIX")

    @staticmethod
    def active_env_name() -> Optional[str]:
        """
        Returns the currently active conda environment name.
        `None` if run outside of a conda environment.
        """
        # we do not check CODNA_DEFAULT_ENV as CONDA_PREFIX is a preferred way
        # to get the active conda environment, e.g./home/$USER/.conda/envs/{env_name}
        env_name: Optional[str] = None
        if env_dir := conda.active_env_dir():
            env_name = os.path.basename(env_dir)

        return env_name
