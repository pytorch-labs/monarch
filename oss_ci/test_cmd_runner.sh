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


# shellcheck source=/dev/null
source "$CONDA_ENV_DIR/$PATH_TO_CONDA_DIR/bin/activate"


if [ -z "$PERF_LOGGER_CMD" ]; then
    echo "$SCRIPT_COMMAND"
    $SCRIPT_COMMAND
else
    # Perf Logger
    TMPFILE_PERF=$(mktemp)
    PERF_STATS_FILE=$TMPFILE_PERF $SCRIPT_COMMAND
    echo "perf state file output"
    cat "$TMPFILE_PERF"; echo

    # shellcheck disable=SC2154
    PERF_STATS_FILE=$TMPFILE_PERF $PERF_LOGGER_CMD
fi
