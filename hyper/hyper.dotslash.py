#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import dotslash

dotslash.export_fbcode_build(
    target="fbcode//monarch/hyper:hyper",
    oncall="monarch",
)
