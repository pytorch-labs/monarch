#!/bin/bash

set -eEx

LIBCUDA="/usr/local/fbcode/platform010/lib/libcuda.so"
if [ -f "$LIBCUDA" ]; then
    export LIBCUDA_DIR="${LIBCUDA%/*}"
    export TRITON_LIBCUDA_PATH="$LIBCUDA_DIR"
    export LD_PRELOAD="$LIBCUDA:/usr/local/fbcode/platform010/lib/libnvidia-ml.so${PRELOAD_PATH:+:$PRELOAD_PATH}"
fi

# Also preload put path to torch libs as for monarch dev workflow we dont
# install it into the env so we need to make sure the binaries can find
# libtorch and friends on mast and the rpaths set during dev install will
# be wrong on mast.
export LD_LIBRARY_PATH="${CONDA_DIR}/lib:${CONDA_DIR}/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$TORCHX_RUN_PYTHONPATH"
export MONARCH_WORKER_MAIN="monarch._monarch.worker.worker_main"

# shellcheck disable=SC1091
source "${CONDA_DIR}/bin/activate"

cd "${WORKSPACE_DIR}"

# Necessary to ensure that the worker has access to the correct
# device. Without it, all workers will try to run on the same gpu.
export CUDA_VISIBLE_DEVICES="$HYPERACTOR_LOCAL_RANK"
# Command to use for spawning pipe processes (see e.g. https://fburl.com/code/6g8uiqk6).
# This should be the same as the command used for spawning workers. Since this script uses
# "python" (as opposed to some XAR binary) to spawn the worker, the value of the env var should be "python".
export MONARCH_WORKER_EXE=python

export HOST_LABEL_NAME="world.monarch.meta.com/host_name"
HOST_NAME="$(hostname)"
export IP_LABEL_NAME="world.monarch.meta.com/ip_addr"
IP=$(hostname -I | cut -d' ' -f1)

exec python -X faulthandler -m "$MONARCH_WORKER_MAIN" worker \
    --world-id "$HYPERACTOR_WORLD_ID" \
    --proc-id "$HYPERACTOR_PROC_ID" \
    --bootstrap-addr "$HYPERACTOR_BOOTSTRAP_ADDR" \
    --extra-proc-labels "$HOST_LABEL_NAME=$HOST_NAME" \
    --extra-proc-labels "$IP_LABEL_NAME=$IP"
