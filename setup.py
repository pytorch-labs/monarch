# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import shutil
import subprocess
import sysconfig

from setuptools import Command, find_packages, setup
from setuptools_rust import Binding, RustExtension

USE_TENSOR_ENGINE = os.environ.get("USE_TENSOR_ENGINE", "1") == "1"


class Clean(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import glob
        import re

        with open(".gitignore") as f:
            ignores = f.read()
            pat = re.compile(r"^#( BEGIN NOT-CLEAN-FILES )?")
            for wildcard in filter(None, ignores.split("\n")):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    # Don't remove absolute paths from the system
                    wildcard = wildcard.lstrip("./")

                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        subprocess.run(["cargo", "clean"])



ENABLE_MSG_LOGGING = (
    "--cfg=enable_hyperactor_message_logging"
    if os.environ.get("ENABLE_MESSAGE_LOGGING")
    else ""
)

# Set environment variables for build
env_vars = {
    "RUSTFLAGS": " ".join(["-Zthreads=16", ENABLE_MSG_LOGGING]),
    "TORCH_SYS_USE_PYTORCH_APIS": "0",
}

# C++ extensions and torch setup (only when tensor engine is enabled)
ext_modules = []
cmdclass = {"clean": Clean}

# Load base requirements
with open("requirements-base.txt") as f:
    reqs = f.read().strip().split("\n")


if USE_TENSOR_ENGINE:
    import torch
    from torch.utils.cpp_extension import (
        BuildExtension,
        CppExtension,
        CUDA_HOME,
        include_paths as torch_include_paths,
        TORCH_LIB_PATH,
    )

    USE_CUDA = CUDA_HOME is not None

    monarch_cpp_src = ["python/monarch/common/init.cpp"]
    if USE_CUDA:
        monarch_cpp_src.append("python/monarch/common/mock_cuda.cpp")

    common_C = CppExtension(
        "monarch.common._C",
        monarch_cpp_src,
        extra_compile_args=["-g", "-O3"],
        libraries=["dl"],
        include_dirs=[
            os.path.dirname(os.path.abspath(__file__)),
            sysconfig.get_config_var("INCLUDEDIR"),
        ],
    )

    controller_C = CppExtension(
        "monarch.gradient._gradient_generator",
        ["python/monarch/gradient/_gradient_generator.cpp"],
        extra_compile_args=["-g", "-O3"],
        include_dirs=[
            os.path.dirname(os.path.abspath(__file__)),
            sysconfig.get_config_var("INCLUDEDIR"),
        ],
    )

    ext_modules = [controller_C, common_C]
    cmdclass["build_ext"] = BuildExtension.with_options(no_python_abi_suffix=True)

    glibcxx_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    env_vars.update({
        "CXXFLAGS": f"-D_GLIBCXX_USE_CXX11_ABI={glibcxx_abi}",
        "LIBTORCH_LIB": TORCH_LIB_PATH,
        "LIBTORCH_INCLUDE": ":".join(torch_include_paths()),
        "_GLIBCXX_USE_CXX11_ABI": str(glibcxx_abi),
    })

    if USE_CUDA:
        env_vars["CUDA_HOME"] = CUDA_HOME

    with open("requirements-tensor.txt") as f:
        tensor_reqs = f.read().strip().split("\n")
    reqs.extend(tensor_reqs)

reqs = "\n".join(reqs)

os.environ.update(env_vars)


with open("README.md", encoding="utf8") as f:
    readme = f.read()

rust_extensions = [
    RustExtension(
        "monarch._rust_bindings",
        binding=Binding.PyO3,
        path="monarch_extension/Cargo.toml",
        debug=False,
        features=["tensor_engine"] if USE_TENSOR_ENGINE else [],
        args=[] if USE_TENSOR_ENGINE else ["--no-default-features"],
    ),
]

if USE_TENSOR_ENGINE:
    rust_extensions.append(
        RustExtension(
            {"controller_bin": "monarch.monarch_controller"},
            binding=Binding.Exec,
            path="controller/Cargo.toml",
            debug=False,
        )
    )

package_name = os.environ.get("MONARCH_PACKAGE_NAME", "monarch")
package_version = os.environ.get("MONARCH_VERSION", "0.0.1")

setup(
    name=package_name,
    version=package_version,
    packages=find_packages(
        where="python",
        exclude=["python/tests.*", "python/tests"],
    ),
    package_dir={"": "python"},
    python_requires=">= 3.10",
    install_requires=reqs.strip().split("\n"),
    license="BSD-3-Clause",
    author="Meta",
    author_email="oncall+monarch@xmail.facebook.com",
    description="Monarch: Single controller library",
    long_description=readme,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    entry_points={
        "console_scripts": [
            "monarch=monarch.tools.cli:main",
            "monarch_bootstrap=monarch.bootstrap_main:invoke_main",
        ],
    },
    rust_extensions=rust_extensions,
    cmdclass=cmdclass,
)
