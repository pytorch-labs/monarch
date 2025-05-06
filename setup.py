import os

import shutil
import subprocess
import sysconfig

import torch

from setuptools import Command, find_packages, setup

from setuptools_rust import Binding, RustExtension
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDA_HOME,
    include_paths as torch_include_paths,
    TORCH_LIB_PATH,
)

common_C = CppExtension(
    "monarch.common._C",
    ["python/monarch/common/mock_cuda.cpp", "python/monarch/common/init.cpp"],
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

ENABLE_MSG_LOGGING = (
    "--cfg=enable_hyperactor_message_logging"
    if os.environ.get("ENABLE_MESSAGE_LOGGING")
    else ""
)

os.environ.update(
    {
        "CXXFLAGS": f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "RUSTFLAGS": " ".join(["-Zthreads=16", ENABLE_MSG_LOGGING]),
        "CUDA_HOME": CUDA_HOME,
        "LIBTORCH_LIB": TORCH_LIB_PATH,
        "LIBTORCH_INCLUDE": ":".join(torch_include_paths()),
        "_GLIBCXX_USE_CXX11_ABI": str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        "TORCH_SYS_USE_PYTORCH_APIS": "0",
    }
)


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


setup(
    name="monarch",
    version="1.0",
    packages=find_packages(
        where="python",
        exclude=["python/tests.*", "python/tests"],
    ),
    package_dir={"": "python"},
    install_requires=[
        "torch",
        "pyzmq",
        "requests",
        "numpy",
        "pyre-extensions",
        "pytest-timeout",
    ],
    author="oncall+monarch@xmail.facebook.com",
    description="Monarch: Single controller library",
    ext_modules=[
        controller_C,
        common_C,
    ],
    rust_extensions=[
        RustExtension(
            "monarch._monarch._lib",
            binding=Binding.PyO3,
            path="monarch_extension/Cargo.toml",
            debug=False,
        ),
        RustExtension(
            {"controller_bin": "monarch.monarch_controller"},
            binding=Binding.Exec,
            path="controller/Cargo.toml",
            debug=False,
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        "clean": Clean,
    },
)
