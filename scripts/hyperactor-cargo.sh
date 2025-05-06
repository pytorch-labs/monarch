#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under both the MIT license found in the
# LICENSE-MIT file in the root directory of this source tree and the Apache
# License, Version 2.0 found in the LICENSE-APACHE file in the root directory
# of this source tree.
#
# TODO: Integrate into CI (example:
# '~/fbsource/tools/utd/migrated_nbtd_jobs/buck2.td')

# Fail if we have any errors
set -e

REPO_DIR=$(cd -P -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && sl root)
FBCODE=$(realpath "$REPO_DIR"/fbcode)
MONARCH_DIR="$FBCODE"/monarch

# Merge stdout and stderr logs because Sandcastle shows them
# separately
exec >&2

# If we fail on CI, better to get as much debug information as we can
export RUST_BACKTRACE=full

# shellcheck source=/dev/null
source "$FBCODE"/monarch/scripts/hyperactor-sandcastle-setup.sh

cd "$MONARCH_DIR"
# Package 'nccl-sys' depends on CUDA. This isn't available in
# Sandcastle so we limit ourselves to the following packages.
packages=(
  "hyperactor_macros"
  "hyperactor"
  "hyperactor_multiprocess"
  "hyperactor_mesh"
  "ndslice"
)
for package in "${packages[@]}"; do
    cargo build --package "$package"

    # TODO: fix me!
    # cargo test --package "$package"
    if [[ "$package" == "hyperactor" ]]; then
        CARGO_TEST=1 cargo test --package hyperactor -- \
              data mailbox parse proc reference actor channel::local
    fi
    if [[ "$package" == "hyperactor_multiprocess" ]]; then
        CARGO_TEST=1 cargo test --package hyperactor_multiprocess --
    fi

    if [[ "$package" == "hyperactor_mesh" ]]; then
        CARGO_TEST=1 cargo test --package hyperactor_mesh --
    fi

    # TODO: not tested yet
    # cargo clippy --package "$package" --all --tests --Dwarnings
done
