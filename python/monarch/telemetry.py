# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import logging

from monarch._rust_bindings.hyperactor_extension.telemetry import (  # @manual=//monarch/monarch_extension:monarch_extension
    forward_to_tracing,
)


class TracingForwarder(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        forward_to_tracing(record)
