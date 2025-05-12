# pyre-unsafe
import os

# Required for OpaqueRef
import monarch.common.tree  # noqa: F401

import torch
from ftar.ftar_py_lib import AllReduceManager, CommOptions, DynamicGroup, WorkerInfo
from monarch.common.opaque_ref import OpaqueRef


_dynamic_group = None


def setup_ftar(global_rank, replica_size, ports, device_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(global_rank)
    dg = DynamicGroup(
        global_rank=global_rank,
        replica_size=replica_size,
        port=ports[0],
        ports=ports,
        device_id=device_id,
    )
    global _dynamic_group
    _dynamic_group = dg
    ar_manager = AllReduceManager(
        global_rank=global_rank,
        dynamic_group=dg,
    )
    return OpaqueRef(ar_manager)


def _to_worker_info(val, ip_addr):
    return WorkerInfo(
        ip_addr=ip_addr,
        port=0,
        ports=list(val[0]),
        rank=val[1],
    )


def reconfig_ftar(ar_manager_ref, quorum, step):
    ar_manager = ar_manager_ref.value
    global _dynamic_group
    ip_addr = _dynamic_group.get_worker_info().ip_addr
    quorum = [_to_worker_info(val, ip_addr) for val in quorum]
    comm_opts = CommOptions(timeout_ms=20_000, qp_connect_timeout_ms=20_000)
    ar_manager.reconfig(
        quorum=quorum, quorum_change_number=step, comm_options=comm_opts
    )
    return torch.tensor(1)


def run_allreduce(ar_manager_ref, tensor_to_reduce):
    allreduce_manager = ar_manager_ref.value
    comm_opts = CommOptions(timeout_ms=10_000)
    work = allreduce_manager.all_reduce(tensor_to_reduce, comm_options=comm_opts)
    # Wait for completion
    success = work.get_result()
    if success:
        # Tensor is allreduced inplace
        return tensor_to_reduce
    raise RuntimeError("AllReduce failed")
