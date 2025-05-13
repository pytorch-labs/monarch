#!/bin/bash

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

CONDA_ENV_PATH="$BUCK_DEFAULT_RUNTIME_RESOURCES/monarch/oss_ci/monarch_conda_env"
conda run -p "$CONDA_ENV_PATH" "$@"
