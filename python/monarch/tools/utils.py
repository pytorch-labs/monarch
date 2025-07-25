# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import os
import pathlib
from typing import Optional


def MONARCH_HOME() -> pathlib.Path:
    """
    Path to the "dot-directory" for monarch.
    Defaults to `~/.monarch` and is overridable via the `MONARCH_HOME` environment variable.
    """

    default_dir = str(pathlib.Path.home() / ".monarch")
    monarch_home = pathlib.Path(os.getenv("MONARCH_HOME", default_dir))
    monarch_home.mkdir(parents=True, exist_ok=True)

    return monarch_home


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
        env_name = os.getenv("CONDA_DEFAULT_ENV")

        if not env_name:
            # conda envs activated with metaconda doesn't set CODNA_DEFAULT_ENV so
            # fallback to CONDA_PREFIX which points to the path of the currently active conda environment
            # e.g./home/$USER/.conda/envs/{env_name}
            if env_dir := conda.active_env_dir():
                env_name = os.path.basename(env_dir)

        return env_name
