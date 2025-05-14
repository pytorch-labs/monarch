"""
TorchX component definition for Monarch server.
Basically a job definition for creating a Monarch server.

See: https://pytorch.org/torchx/main/basics.html#components
"""

# pyre-strict
import os
from typing import Dict, List, NamedTuple, Optional

import torchx.specs as specs
from monarch.tools.components import base
from monarch.tools.mesh_spec import DEFAULT_REMOTE_ALLOCATOR_PORT


_DEFAULT_ENV: dict[str, str] = {
    # --- ftar configuration ---
    "FTAR_COORDINATOR": "/packages/ftar_core_lib/coordinator_sidecar",
    "FTAR_LISTENER_SOCK": "/tmp/ftar_coord.sock",
    "FTAR_LISTENER_PORT": "18713",
    # --- nvidia libs configuration ---
    "NVTE_TORCH_COMPILE": "0",
    "NVTE_BIAS_GELU_NVFUSION": "0",
    "NVTE_CUDA_INCLUDE_DIR": "/usr/local/cuda/include",
    "NVTE_DISABLE_NVRTC": "1",
    "NVTE_FUSED_ATTN": "1",
    "NVTE_FUSED_ATTN_USE_FAv2_BWD": "1",
    "NCCL_SET_THREAD_NAME": "1",
    "NCCL_DEBUG_SUBSYS": "INIT,COLL,P2P,SHM,NET,GRAPH,TUNING,ENV,ALLOC",
    "NCCL_ASYNC_ERROR_HANDLING": "3",
    "NCCL_NET_OVERHEAD": "2750",
    "NCCL_IB_SPLIT_DATA_ON_QPS": "0",
    "NCCL_IB_QPS_PER_CONNECTION": "16",
    "NCCL_CTRAN_ENABLE": "0",
    # --- torch configuration ---
    "TORCH_SHOW_CPP_STACKTRACES": "1",
    "PYTORCH_JIT": "0",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "GLOG_minloglevel": "1",
    # --- xlformers configuration ---
    "XLFORMERS_ON_DEVGPU_RTPTEST": "1",
    # -- fs configuration --
    "DISABLE_NFS": "1",  # NFS deprecated behind justknobs; disable it explicitly just in case
    "FUSE_DST": "/mnt/wsfuse",  # required for DUMP_DIR
    "FUSE_ENABLE_OVERWRITES": "1",
    "OILFS_EXTRA_FLAGS_GENAI": "--oilfs_cto_periodic_refresh=15s",
    # --- airstore configuration ---
    "ENABLE_AIRSTORE": "0",
    "AIRSTORE_DECRYPT_SERVER_PATH": "/packages/ws_airstore.client/decrypt_server",
    "AIRSTORE_LOCAL_MOUNT_ROOT": "/mnt/airstore",
    # --- enable_ttls configuration ---
    "https_proxy": "http://fwdproxy:8080",
    "http_proxy": "http://fwdproxy:8080",
    "no_proxy": (
        ".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,"
        ".fburl.com,.facebook.net,.sb.fbsbx.com,localhost"
    ),
}

_SYSTEMD_SERVICES = {
    # service_name: should_activate
    "snapshot_code": False,
    "genai_agent": False,
    "paft_zelos_coordinator": False,
}


_DEFAULT_MESHES = ["mesh_0:1:gtt_any"]


class MeshSpec(NamedTuple):
    name: str
    num_hosts: int
    host_type: str


def hyperactor(
    # TODO kiuk@ figure out a better way to pass mesh specs
    #  right now TorchX component function signature is limited to
    #  primitives and list, dict of primitives so we encode
    #  mesh specs as ':' delimited str for now
    meshes: List[str] = _DEFAULT_MESHES,
    env: Optional[Dict[str, str]] = None,
    port: int = DEFAULT_REMOTE_ALLOCATOR_PORT,
    hyperactor_fbpkg: str = "monarch:prod",
    program: str = f"{specs.macros.img_root}/projects/monarch/scripts/run_monarch_bootstrap.sh",
    systemd_services: Optional[Dict[str, bool]] = None,
    dump_dir_id: str = "${app_id}",
) -> specs.AppDef:
    """Creates a Monarch server on hyperactor per the given parameters.

    Args:
        meshes: list of mesh specs of the form "{name}:{num_hosts}:{host_type}"
    """

    systemd_services = dict(_SYSTEMD_SERVICES) | (systemd_services or {})
    env = dict(_DEFAULT_ENV) | (env or {})

    # fbpkgs to download
    packages = base.Packages()
    packages.add_package("oil.oilfs:stable")
    packages.add_package("conda_mast_core:stable")
    packages.add_package("fb-py-spy:prod")
    packages.add_shared_lib("folly.symbolizer:prod", "libFollySegFault.so")
    packages.add_shared_lib("ttls_so:stable", "TransparentTls3.so")

    if env["ENABLE_AIRSTORE"]:
        packages.add_python_lib("ws_airstore.client:stable")

    if systemd_services.get("paft_zelos_coordinator", False):  # enable_ftar
        packages.add_python_lib("ftar_core_lib:stable")

    env["TORCHX_RUN_PYTHONPATH"] = packages.PYTHONPATH
    env["PRELOAD_PATH"] = packages.PRELOAD_PATH

    env["XLF_SYSTEMD_SERVICES"] = ",".join(
        [
            service
            for service, should_activate in systemd_services.items()
            if should_activate
        ]
    )

    dump_mount = env["FUSE_DST"]
    # Make the dump dir available for shell scripts
    env["DUMP_DIR"] = os.path.join(dump_mount, "outputs", dump_dir_id)

    # core and mesh task groups have the same entrypoint
    # TODO kiuk@ make torchx support chained entrypoints
    entrypoint = " ".join(
        [
            # can't use specs.macros.img_root in entrypoint (torchx only resolves macros in args, env, metadata)
            "/packages/conda_mast_core/mount/mount.sh;",
            "$WORKSPACE_DIR/tools/launching/torchx/entrypoint/systemd_launcher.sh &&",
            "$WORKSPACE_DIR/tools/launching/torchx/entrypoint/rank_assignment.sh",  # TODO maybe not needed?
        ]
    )

    return base.hyperactor(
        meshes=meshes,
        env=env,
        port=port,
        hyperactor_fbpkg=hyperactor_fbpkg,
        pre_launch_cmd=entrypoint,
        additional_packages=packages,
        program=program,
        systemd_services=systemd_services,
    )
