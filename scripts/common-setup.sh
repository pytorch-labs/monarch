#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Common setup functions for Monarch CI workflows

set -ex

# Setup conda environment. Defaults to Python 3.10.
setup_conda_environment() {
    local python_version=${1:-3.10}
    echo "Setting up conda environment with Python ${python_version}..."
    conda create -n venv python="${python_version}" -y
    conda activate venv
    export PATH=/opt/rh/devtoolset-10/root/usr/bin/:$PATH
    python -m pip install --upgrade pip
}

# Install system-level dependencies
install_system_dependencies() {
    echo "Installing system dependencies..."
    dnf update -y
    dnf install clang-devel libunwind libunwind-devel -y
}

# Install and configure Rust nightly toolchain
setup_rust_toolchain() {
    echo "Setting up Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "${HOME}"/.cargo/env
    rustup toolchain install nightly
    rustup default nightly
}

install_build_dependencies() {
    echo "Installing build dependencies..."
    pip install -r build-requirements.txt
}

# Install Python test dependencies
install_python_test_dependencies() {
    echo "Installing test dependencies..."
    pip install -r python/tests/requirements.txt
}

# Install wheel from artifact directory
install_wheel_from_artifact() {
    echo "Installing wheel from artifact..."
    pip install "${RUNNER_ARTIFACT_DIR}"/*.whl
}

# Build process allocator binary
build_process_allocator() {
    echo "Building process allocator binary..."
    cargo build --manifest-path monarch_hyperactor/Cargo.toml --release
    mkdir -p "${RUNNER_ARTIFACT_DIR}"/cargo_bin
    mv target/release/process_allocator "${RUNNER_ARTIFACT_DIR}"/cargo_bin
}


# Common setup for build workflows (environment + system deps + rust)
setup_build_environment() {
    local python_version=${1:-3.10}
    setup_conda_environment "${python_version}"
    install_system_dependencies
    setup_rust_toolchain
    install_build_dependencies
}

# Common setup for test workflows (environment only)
setup_test_environment() {
    local python_version=${1:-3.10}
    setup_conda_environment "${python_version}"
    install_python_test_dependencies
}
