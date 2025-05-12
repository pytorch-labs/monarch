# pyre-unsafe
from pathlib import Path

import fsspec
from fsspec.implementations.local import LocalFileSystem
from monarch import IN_PAR


_root = Path(__file__).parent
test_tokenizer = str(_root / "test_tiktoken.model")
test_dataset = str(_root / "c4_test")


class LocalFileSystemThatTricksHuggingFaceDatasetsIntoLoadingSymlinksBecauseBuckCanOnlyCreateSymLinkResources(
    LocalFileSystem
):
    def glob(self, *args, **kwargs):
        result = super().glob(*args, **kwargs)
        if isinstance(result, dict):
            for info in result.values():
                if info["type"] == "other":
                    info["type"] = "file"
        return result


if IN_PAR:
    fsspec.register_implementation(
        "file",
        LocalFileSystemThatTricksHuggingFaceDatasetsIntoLoadingSymlinksBecauseBuckCanOnlyCreateSymLinkResources,
        clobber=False,
    )
