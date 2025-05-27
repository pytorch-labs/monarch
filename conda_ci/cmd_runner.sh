#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex


LIBCUDA="/usr/local/fbcode/platform010/lib/libcuda.so"
if [ -f "$LIBCUDA" ]; then
    export LIBCUDA_DIR="${LIBCUDA%/*}"
    export TRITON_LIBCUDA_PATH="$LIBCUDA_DIR"
    export LD_PRELOAD="$LIBCUDA:/usr/local/fbcode/platform010/lib/libnvidia-ml.so${PRELOAD_PATH:+:$PRELOAD_PATH}"
fi

MCP_AUTO=0
NCCL_IGNORE_TOPO_LOAD_FAILURE=true

echo "$NCCL_IGNORE_TOPO_LOAD_FAILURE"
echo "$MCP_AUTO"

CONDA_ENV_PATH="$BUCK_DEFAULT_RUNTIME_RESOURCES/monarch/conda_ci/monarch_conda_env"
conda run -p "$CONDA_ENV_PATH" "$@"
