use std::time::UNIX_EPOCH;

use opentelemetry_sdk::Resource;
use tracing_core::Metadata;
use tracing_core::Subscriber;
use tracing_core::field;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::registry::SpanRef;

use crate::meta::scuba_utils;
use crate::meta::scuba_utils::generate_scuba_data;

pub struct Layer {
    scuba: scuba_utils::ScubaClientWrapper,
    resource: Resource,
}

impl Layer {
    pub fn builder(resource: Resource) -> LayerBuilder {
        LayerBuilder::new(resource)
    }
}

pub struct LayerBuilder {
    resource: Resource,
}

impl LayerBuilder {
    fn new(resource: Resource) -> Self {
        Self { resource }
    }

    pub fn build(self) -> Layer {
        // TODO: implement the build method to construct a Layer instance using the provided resource
        Layer {
            resource: self.resource.clone(),
            scuba: generate_scuba_data(&self.resource),
        }
    }
}

struct SampleRecorder(scuba::ScubaSample);

impl<'a, S> From<&SpanRef<'a, S>> for SampleRecorder
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    fn from(span: &SpanRef<'a, S>) -> Self {
        let mut sample: SampleRecorder = span.metadata().into();
        let id = span.id();
        sample.0.add("span_id", id.into_u64() as i64);
        if let Some(parent) = span.parent() {
            sample
                .0
                .add("parent_span_id", parent.id().into_u64() as i64);
        }
        let now = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap();
        sample.0.add("time", now.as_secs() as i64);
        sample.0.add("time_us", now.as_micros() as i64);
        sample
    }
}

impl From<&Metadata<'static>> for SampleRecorder {
    fn from(meta: &Metadata<'static>) -> Self {
        let mut sample = SampleRecorder(scuba::ScubaSample::new());
        sample.0.add("level", meta.level().as_str());
        sample.0.add("loc.file", meta.file().unwrap_or_default());
        sample
            .0
            .add("loc.line", meta.line().unwrap_or_default() as i64);
        sample.0.add("target", meta.target());
        sample
    }
}
impl field::Visit for SampleRecorder {
    fn record_debug(&mut self, field: &field::Field, value: &dyn std::fmt::Debug) {
        self.0.add(field.name(), format!("{:?}", value).as_str());
    }

    fn record_f64(&mut self, field: &field::Field, value: f64) {
        self.0.add(field.name(), value);
    }

    fn record_i64(&mut self, field: &field::Field, value: i64) {
        self.0.add(field.name(), value);
    }

    fn record_u64(&mut self, field: &field::Field, value: u64) {
        self.0.add(field.name(), value as i64);
    }

    fn record_bool(&mut self, field: &field::Field, value: bool) {
        self.0.add(field.name(), value as i64);
    }

    fn record_str(&mut self, field: &field::Field, value: &str) {
        self.0.add(field.name(), value);
    }
}

impl<S> tracing_subscriber::Layer<S> for Layer
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    fn on_new_span(
        &self,
        attrs: &tracing_core::span::Attributes<'_>,
        id: &tracing_core::span::Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if let Some(span) = ctx.span(id) {
            let mut sample: SampleRecorder = (&span).into();
            attrs.record(&mut sample);
            let mut sample = sample.0;
            if let Some(parent) = span.parent() {
                sample.add("parent_span_id", parent.id().into_u64() as i64);
            }
            sample.add("event_type", "start_span");
            self.scuba.log(sample);
        }
    }

    fn on_event(
        &self,
        event: &tracing_core::Event<'_>,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut sample = SampleRecorder::from(event.metadata()).0;
        if let Some(id) = event.parent() {
            if let Some(span) = ctx.span(id) {
                sample.add("span_id", id.into_u64() as i64);
                if let Some(parent) = span.parent() {
                    sample.add("parent_span_id", parent.id().into_u64() as i64);
                }
            }
        }
        let now = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap();
        sample.add("time", now.as_secs() as i64);
        sample.add("time_us", now.as_micros() as i64);
        sample.add("event_type", "instant_event");
        let mut sample = SampleRecorder(sample);
        event.record(&mut sample);
        self.scuba.log(sample.0);
    }

    fn on_close(&self, id: tracing_core::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(span) = ctx.span(&id) {
            let mut sample = SampleRecorder::from(span.metadata()).0;
            sample.add("name", span.name());
            sample.add("span_id", id.into_u64() as i64);
            if let Some(parent) = span.parent() {
                sample.add("parent_span_id", parent.id().into_u64() as i64);
            }
            let now = std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap();
            sample.add("time", now.as_secs() as i64);
            sample.add("time_us", now.as_micros() as i64);
            sample.add("event_type", "end_span");
            self.scuba.log(sample);
        }
    }
}
