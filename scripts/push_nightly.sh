#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Make this script executable with: chmod +x scripts/push_nightly.sh

set -ex

# Get the current date in YYYYMMDD format
DATE=$(date +%Y%m%d)

# Set the package name with the date suffix
export MONARCH_PACKAGE_NAME="torchmonarch-nightly"
export MONARCH_VERSION="0.0.1.dev${DATE}"

# Build the wheel
python setup.py bdist_wheel

# Upload to PyPI using twine
python -m twine upload \
  --username __token__ \
  --password "${SECRET_PYPI_TOKEN}" \
  dist/monarch_nightly-*-py3-none-any.whl
