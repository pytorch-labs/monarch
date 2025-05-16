#!/bin/bash

# Runner script that allows main script to be run on RE machines.

LIBCUDA="/usr/local/fbcode/platform010/lib/libcuda.so"
if [ -f "$LIBCUDA" ]; then
    export LIBCUDA_DIR="${LIBCUDA%/*}"
    export TRITON_LIBCUDA_PATH="$LIBCUDA_DIR"
    export LD_PRELOAD="$LIBCUDA:/usr/local/fbcode/platform010/lib/libnvidia-ml.so${PRELOAD_PATH:+:$PRELOAD_PATH}"
fi

MCP_AUTO=0
echo "$MCP_AUTO"
# shellcheck source=/dev/null
source conda_env/conda/bin/activate
python "$@"
