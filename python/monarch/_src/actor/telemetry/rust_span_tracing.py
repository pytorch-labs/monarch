# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from contextlib import contextmanager
from typing import Iterator, Mapping, Optional, Union

import opentelemetry.util.types as types  # @manual=fbsource//third-party/pypi/opentelemetry-api:opentelemetry-api
from monarch._rust_bindings.monarch_hyperactor.telemetry import (
    get_current_span_id,
    PySpan,
)
from opentelemetry import (  # @manual=fbsource//third-party/pypi/opentelemetry-api:opentelemetry-api
    trace,
)
from opentelemetry.trace.status import Status, StatusCode

logger: logging.Logger = logging.getLogger(__name__)


class SpanWrapper(trace.Span):
    def __init__(self, name: str, actor_id: Optional[str]) -> None:
        super().__init__()
        self._span: PySpan | None = PySpan(name, actor_id)

    def end(self, end_time: Optional[int] = None) -> None:
        # since PySpan is not sendable, we need to make sure it is deallocated on this thread so it doesn't log warnings.
        s = self._span
        assert s is not None
        s.exit()
        self._span = None
        del s

    def record_exception(
        self,
        exception: BaseException,
        attributes: types.Attributes = None,
        timestamp: Optional[int] = None,
        escaped: bool = False,
    ) -> None:
        pass

    def is_recording(self) -> bool:
        return False

    def get_span_context(self) -> trace.span.SpanContext:
        span_id = get_current_span_id()
        return trace.span.SpanContext(trace_id=0, span_id=span_id, is_remote=False)

    def set_attributes(self, attributes: Mapping[str, types.AttributeValue]) -> None:
        pass

    def set_attribute(self, key: str, value: types.AttributeValue) -> None:
        pass

    def add_event(
        self,
        name: str,
        attributes: types.Attributes = None,
        timestamp: Optional[int] = None,
    ) -> None:
        pass

    def update_name(self, name: str) -> None:
        pass

    def set_status(
        self,
        status: Union[Status, StatusCode],
        description: Optional[str] = None,
    ) -> None:
        pass


class RustTracer(trace.Tracer):
    def start_span(
        self,
        name: str,
        context: Optional[trace.Context] = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
        attributes: types.Attributes = None,
        links: trace._Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> trace.Span:
        actor_id = str(attributes.get("actor_id")) if attributes else None
        return SpanWrapper(name, actor_id)

    @contextmanager
    # pyre-fixme[15]: `start_as_current_span` overrides method defined in `Tracer`
    #  inconsistently.
    def start_as_current_span(
        self,
        name: str,
        context: Optional[trace.Context] = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
        attributes: types.Attributes = None,
        links: trace._Links = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[trace.Span]:
        actor_id = str(attributes.get("actor_id")) if attributes else None
        with SpanWrapper(name, actor_id) as s:
            with trace.use_span(s):
                yield s
            del s


class RustTracerProvider(trace.TracerProvider):
    def get_tracer(
        self,
        instrumenting_module_name: str,
        *args: object,
        instrumenting_library_version: Optional[str] = None,
        schema_url: Optional[str] = None,
        **kwargs: object,
    ) -> trace.Tracer:
        return RustTracer()
