# pyre-strict
import logging
import subprocess
import sys
from typing import Optional

from monarch_supervisor.logging import gethostname, initialize_logging

pid: str
logger: logging.Logger = logging.getLogger(__name__)


def extract_pss(pid: str) -> Optional[str]:
    try:
        with open(f"/proc/{pid}/smaps_rollup", "r") as f:
            for line in f.readlines():
                if line.startswith("Pss:"):  # Check if the line starts with 'Pss:'
                    return " ".join(line.split()[1:3])
    except Exception:
        pass
    return None


def log_pstree_output(pid: int) -> None:
    pstree_output = subprocess.check_output(["pstree", "-Tap", str(pid)]).decode(
        "utf-8"
    )
    lines = pstree_output.split("\n")
    logger.info("Process Info")
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(",")
        pids = parts[1].split()[0]
        mem = extract_pss(pids)
        logger.info(f"{line} {mem}")


if __name__ == "__main__":
    (pid,) = sys.argv[1:]
    initialize_logging(f"{gethostname()} host-manager")
    log_pstree_output(int(pid))
