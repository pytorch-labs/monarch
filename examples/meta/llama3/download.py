import os
from typing import Optional

from requests.exceptions import HTTPError


def hf_download(repo_id: Optional[str] = None, hf_token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download

    os.makedirs(f"checkpoints/{repo_id}", exist_ok=True)
    try:
        snapshot_download(
            repo_id,
            local_dir=f"checkpoints/{repo_id}",
            local_dir_use_symlinks=False,
            token=hf_token,
        )
    except HTTPError as e:
        if e.response.status_code == 401:
            print(
                "You need to pass a valid `--hf_token=...` to download private checkpoints."
            )
        else:
            raise e


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download data from HuggingFace Hub.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Repository ID to download from.",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="HuggingFace API token."
    )

    args = parser.parse_args()
    hf_download(args.repo_id, args.hf_token)
