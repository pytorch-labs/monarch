#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under both the MIT license found in the
# LICENSE-MIT file in the root directory of this source tree and the Apache
# License, Version 2.0 found in the LICENSE-APACHE file in the root directory
# of this source tree.

# Set up sandcastle for building monarch

# Fail if we have any errors
set -e
set -o pipefail

# Make sure we can get to the internet, but don't use a proxy for RE/CAS
export HTTPS_PROXY=fwdproxy:8080
export NO_PROXY=.internal.tfbnw.net,interngraph.scgraph.facebook.com,interngraph.intern.facebook.com

# Make cargo fetching for git use the git tool. Works either way on Linux,
# but on Windows this variable is required for fetching with our proxy.
export CARGO_NET_GIT_FETCH_WITH_CLI=true

# On Windows we have high numbers of failure to check the revocation server,
# and this the recommended fix https://github.com/rust-lang/cargo/issues/7096
if [ "$OS" == "Windows_NT" ]; then
    export CARGO_HTTP_CHECK_REVOKE=false
fi

# Install Rust nightly
echo "Downloading rustup and intalling rust toolchain..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain=none --no-modify-path
echo "Toolchain installed."

# Make sure Rust is on the PATH
if [ "$OS" == "Windows_NT" ]; then
    PATH="$(cygpath "$USERPROFILE/.cargo/bin"):$PATH"
    export PATH
else
    export PATH="$HOME/.cargo/bin:$PATH"
fi

if [ "$OS" == "Windows_NT" ]; then
    # Set out-dirs to point to scratch drive D which is faster and larger
    # See: https://fb.workplace.com/groups/341156780065932/permalink/1345911959590404/
    export TMP="D:\\"
    export CARGO_TARGET_DIR="D:\\cargo-target-ci"
else
    # Sometimes we get errors like:
    #   error: failed to run custom build command for `crossbeam-utils v0.7.0`
    #   could not execute process `/data/sandcastle/boxes/eden-trunk-hg-fbcode-fbsource/fbcode/buck2/target/debug/build/crossbeam-utils-d8824d6c0d209c1f/build-script-build` (never executed)
    #   Text file busy (os error 26)
    # We assume that is Eden getting in the way, so put the target directory under $HOME in CI
    export CARGO_TARGET_DIR=$HOME/cargo-target-ci
fi
