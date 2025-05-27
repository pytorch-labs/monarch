#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pepares tmp dir to be uploaded to RE
# Creates ./conda_env dir with conda environment
# Creates ./conda_ci dir with necessary runner scripts
# Creates ./source dir with main script

if [ -z "${TMP_DIR}" ]; then
  # Create a temporary file
  TMP_DIR=$(mktemp -d)
fi

TMP_CONDA_ENV_DIR="$TMP_DIR/conda_env"
TMP_OSS_CI_DIR="$TMP_DIR/conda_ci"
TMP_SOURCE_DIR="$TMP_DIR/source"


mkdir -p "$TMP_CONDA_ENV_DIR"
mkdir -p "$TMP_OSS_CI_DIR"
mkdir -p "$TMP_SOURCE_DIR"

cp -r "$CONDA_DIR"/* "$TMP_CONDA_ENV_DIR"
cp -r "$OSS_CI_DIR"/* "$TMP_OSS_CI_DIR"
cp "$1" "$TMP_SOURCE_DIR/script.py"

echo "Copied files to $TMP_DIR"
