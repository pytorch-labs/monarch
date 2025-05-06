import getpass
import logging
import os
import re
import socket
from functools import lru_cache
from pathlib import Path
from typing import Dict, Final, Mapping, Set

USER = getpass.getuser()

ROOT_DUMP_DIR = {
    "h2": f"/checkpoint/{USER}/xldumps",
    "aws": "/fsx/guismay/xldumps",
    "azure": "/data/side/xldumps",
    "rsc": "/checkpoint/fair_llm/xldumps",
    "devgpu_eag": f"/tmp/xlformers/{USER}/xldumps",
    "devgpu_pci": f"/tmp/xlformers/{USER}/xldumps",
    "devgpu_gtn": f"/tmp/xlformers/{USER}/xldumps",
    "devgpu_rva": f"/tmp/xlformers/{USER}/xldumps",
    "devgpu_cln": f"/tmp/xlformers/{USER}/xldumps",
    "devgpu_ftw": f"/tmp/xlformers/{USER}/xldumps",
    "mast": "/mnt/wsfuse/outputs/",
    "mast_nfs": "/mnt/gen_ai_input_data_nfs/outputs/",
    "environ": os.path.join(
        os.environ.get("XLFORMERS_LARGE_EXPERIMENTS_PATH", ""), "xldumps"
    ),
}


@lru_cache()
def get_cluster() -> str:
    if "XLFORMERS_LARGE_EXPERIMENTS_PATH" in os.environ:
        return "environ"
    hostname = socket.gethostname().split(".")
    devgpu = (
        len(hostname) == 4
        and hostname[0].startswith("devgpu")
        and hostname[2] == "facebook"
        and hostname[3] == "com"
    )
    cluster = {
        "h2": os.path.exists("/private/home") and os.path.exists("/large_experiments/"),
        "aws": os.path.exists("/fsx-llm"),
        "azure": os.path.exists("/data") and os.path.exists("/shared/home"),
        "rsc": os.path.exists("/checkpoint/fair_llm/"),
        "devgpu_ash": devgpu and hostname[1].startswith("ash"),
        "devgpu_eag": devgpu and hostname[1].startswith("eag"),
        "devgpu_pci": devgpu and hostname[1].startswith("pci"),
        "devgpu_gtn": devgpu and hostname[1].startswith("gtn"),
        "devgpu_rva": devgpu and hostname[1].startswith("rva"),
        "devgpu_cln": devgpu and hostname[1].startswith("cln"),
        "devgpu_ftw": devgpu and hostname[1].startswith("ftw"),
        "mast": "MAST_ENVIRONMENT" in os.environ
        and os.environ.get("NFS_DUMP_READ", "0") != "1",
        "mast_nfs": "MAST_ENVIRONMENT" in os.environ
        and os.environ.get("NFS_DUMP_READ", "0") == "1",
    }
    where = [k for k, v in cluster.items() if v]
    if len(where) != 1:
        raise RuntimeError(f"Could not determine current cluster: {cluster}")
    return where[0]


_LARGE_EXPERIMENT_CLUSTER_MAP: Final[Mapping[str, str]] = {
    "h2": "/large_experiments/fair_llm/datasets",
    "h2_old": "/large_experiments/theorem/datasets",
    "aws": "/fsx/guismay/data/large_experiments/fair_llm/datasets",
    "azure": "/data/side/marmot/large_experiments/fair_llm/datasets",
    "rsc": "/checkpoint/fair_llm/theorem/data/large_experiments/theorem/datasets",
    "rsc_old": "/checkpoint/theorem/data/large_experiments/theorem/datasets",
    "devgpu_ash": f"/home/{USER}/eag-wsf/fair_llm_v2/datasets",
    "devgpu_eag": f"/home/{USER}/eag-wsf/fair_llm_v2/datasets",
    "devgpu_gtn": f"/home/{USER}/pci-wsf/fair_llm_v2/datasets",
    "devgpu_pci": f"/home/{USER}/pci-wsf/fair_llm_v2/datasets",
    "devgpu_rva": f"/home/{USER}/pci-wsf/fair_llm_v2/datasets",
    "devgpu_ftw": f"/home/{USER}/pci-wsf/fair_llm_v2/datasets",
    "mast": "/mnt/wsfuse/fair_llm_v2/datasets",
    "mast_nfs": "/mnt/gen_ai_input_data_nfs/fair_llm_v2/datasets",
}


def clusterify_large_experiments_dataset(path: str) -> str:
    return _clusterify(
        path,
        _LARGE_EXPERIMENT_CLUSTER_MAP
        | {
            "environ": os.environ.get(
                "XLFORMERS_LARGE_EXPERIMENTS_PATH", "<NOT_DEFINED>"
            )
        },
    )


_SHUFFLED_CLUSTER_MAP: Final[Mapping[str, str]] = {
    "rsc_old": "/checkpoint/theorem/data/shuffled",
    "h2": "/large_experiments/fair_llm/data/shuffled",
    "azure": "/data/side/marmot/shuffled",
    "rsc": "/checkpoint/fair_llm/theorem/data/shuffled",
    "devgpu_eag": f"/home/{USER}/eag-wsf/fair_llm_v2/shuffled",
    "devgpu_pci": f"/home/{USER}/pci-wsf/fair_llm_v2/shuffled",
    "devgpu_gtn": f"/home/{USER}/pci-wsf/fair_llm_v2/shuffled",
    "devgpu_rva": f"/home/{USER}/pci-wsf/fair_llm_v2/shuffled",
    "devgpu_ftw": f"/home/{USER}/pci-wsf/fair_llm_v2/shuffled",
    "mast": "/mnt/wsfuse/fair_llm_v2/shuffled",
    "mast_nfs": "/mnt/gen_ai_input_data_nfs/fair_llm_v2/shuffled",
}


def clusterify_shuffled(path: str) -> str:
    """
    Update paths for multi iterator.
    """
    return _clusterify(
        path,
        _SHUFFLED_CLUSTER_MAP
        | {
            "environ": os.environ.get(
                "XLFORMERS_LARGE_EXPERIMENTS_PATH", "<NOT_DEFINED>"
            )
        },
    )


_TOKENIZER_CLUSTER_MAP: Final[Mapping[str, str]] = {
    "rsc_old": "/checkpoint/theorem/data/tokenizers",
    "h2": "/large_experiments/fair_llm/data/tokenizers",
    "h2_checkpoints": "/checkpoint/fair_llm/data/tokenizers",
    "aws": "/fsx-llm/shared/checkpoints/tokenizers",
    "azure": "/data/side/marmot/tokenizers",
    "rsc": "/checkpoint/fair_llm/theorem/data/tokenizers",
    "devgpu_ash": f"/home/{USER}/eag-wsf/tokenizers",
    "devgpu_eag": f"/home/{USER}/eag-wsf/tokenizers",
    "devgpu_pci": f"/home/{USER}/pci-wsf/tokenizers",
    "devgpu_gtn": f"/home/{USER}/pci-wsf/tokenizers",
    "devgpu_rva": f"/home/{USER}/pci-wsf/tokenizers",
    "devgpu_ftw": f"/home/{USER}/pci-wsf/tokenizers",
    "mast": "/mnt/wsfuse/tokenizers",
    "mast_nfs": "/mnt/gen_ai_input_data_nfs/tokenizers",
}


def clusterify_tokenizer(path: str) -> str:
    """
    Update paths for multi iterator.
    """
    return _clusterify(
        path,
        _TOKENIZER_CLUSTER_MAP
        | {
            "environ": os.environ.get(
                "XLFORMERS_LARGE_EXPERIMENTS_PATH", "<NOT_DEFINED>"
            )
        },
    )


def _clusterify(path: str, cluster_to_root: Dict[str, str]) -> str:
    path = re.sub(r"/+", "/", path)

    src_clusters = [
        c for c, root in cluster_to_root.items() if path.startswith(_tailing(root))
    ]
    if len(src_clusters) == 0 or os.environ.get("SKIP_CLUSTERIFY", "0") == "1":
        # no match
        return path

    src_roots: Set[str] = {cluster_to_root[src_cluster] for src_cluster in src_clusters}
    # if it is the case we should check
    # that the root of these clusters are the same
    assert len(src_roots) == 1, (
        f"Found more than one possible root, path: {path}, "
        f"source clusters: {src_clusters} "
        f"source roots: {src_roots}"
    )

    (src_root,) = src_roots
    src_root = _tailing(src_root)
    assert src_root.endswith("/")
    suffix = path[len(src_root) :]

    tgt_cluster = get_cluster()
    if tgt_cluster not in cluster_to_root:
        raise RuntimeError(f"Unexpected cluster: {tgt_cluster} to clusterify {path}")
    tgt_root = _tailing(cluster_to_root[tgt_cluster])
    assert tgt_root.endswith("/")

    return f"{tgt_root}{suffix}"


def _tailing(p: str, tail: str = "/") -> str:
    if not p.endswith(tail):
        return f"{p}{tail}"
    return p


def clusterify_data_path(path: str) -> str:
    path = path.strip()
    before = path
    path = clusterify_large_experiments_dataset(path)
    path = clusterify_shuffled(path)
    path = clusterify_tokenizer(path)
    if path != before:
        logging.info(f"Clusterified {before} in {path}")

    return path


def get_root_dump_dir() -> str:
    """Root folder where experiments are stored."""
    root_dir = ROOT_DUMP_DIR[get_cluster()]
    if get_cluster() == "h2":
        Path(root_dir).mkdir(exist_ok=True, parents=True)
    assert os.path.isdir(root_dir), root_dir
    return root_dir


def get_personal_root_dump_dir() -> str:
    root_dir = get_root_dump_dir()
    if get_cluster() == "h2":
        # on H2, root_dir is already personalized
        personal_root_dir = Path(root_dir)
    else:
        personal_root_dir = Path(root_dir) / "xldumps"
    personal_root_dir.mkdir(exist_ok=True, parents=True)
    return str(personal_root_dir)


def get_job_id() -> str:
    if os.environ.get("SLURM_JOB_ID"):
        return os.environ.get("SLURM_JOB_ID", "")
    if os.environ.get("MAST_HPC_JOB_NAME"):
        return os.environ.get("MAST_HPC_JOB_NAME", "")
    return "0"


def get_restart_index() -> str:
    if os.environ.get("SLURM_RESTART_COUNT"):
        return os.environ.get("SLURM_RESTART_COUNT", "")
    if os.environ.get("MAST_HPC_JOB_ATTEMPT_INDEX"):
        version = os.environ.get("MAST_HPC_JOB_VERSION", "")
        mast_attempt = os.environ.get("MAST_HPC_JOB_ATTEMPT_INDEX", "")
        return f"v{version}_attempt{mast_attempt}"
    return "0"


if __name__ == "__main__":
    print(f"I'm on {get_cluster()} !")
