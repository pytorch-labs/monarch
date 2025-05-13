mod metrics;
mod scuba_tracing;
mod scuba_utils;
use crate::key_value;
use crate::kv_pairs;

fn pairs_from_env<'a, I: IntoIterator<Item = &'a str>>(var_names: I) -> Vec<KeyValue> {
    var_names
        .into_iter()
        .filter_map(|name| match std::env::var(name) {
            Ok(val) => Some((name, val)),
            Err(_) => None,
        })
        .map(|(key, val)| KeyValue::new(key.to_ascii_lowercase(), val))
        .collect()
}

pub fn meter_provider() -> opentelemetry_sdk::metrics::SdkMeterProvider {
    metrics::MeterProviderBuilder::new(metrics_resource()).build()
}

use opentelemetry::Array;
use opentelemetry::KeyValue;
use opentelemetry::StringValue;
use opentelemetry::Value;
use opentelemetry_sdk::Resource;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::filter::Targets;

use super::env::Env;
/// High level overview of how we configure exporting of logs and metrics to scuba.
/// We use opentelemetry and the tracing library as our "API" that is OSS friendly. We then
/// have an implementation of "backends" for these APIs that writes the various data to scuba.
///
/// Our sdk is configured by a few special attributes
///
/// 'meta.scuba.table'   -> The target scuba table to log to (required)
/// 'meta.scuba.subset'  -> The subset of that scuba table to log to (dev/prod/test) (optional)
/// 'meta.scuba.columns' -> The columns you want included on every scuba sample, requredlesss of if they are manually set.
///                       This is useful for say, including the execution_id on every sample to every table.
///                       The elements of this array MUST match the name of attributes set on the same resource.
///                       If the resource does not containe a key with this name, it will be ignored.
///                       For storage reasons, we try to put all "global" and static information in the executions table
///                       so that we can join against it at query time without filling our other tables with redundant information.
///
/// You can put whatever other attributes you like on the resource. They will not hurt anything and may be refered to by the 'meta.scuba.columns' attribute.
// will default to submitting metrics every 60s. Can override this with the env vars
// outlined https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/#periodic-exporting-metricreader
/// Primarally, these APIs are configured using an [`opentelemetry_sdk::Resource`]
/// Initializes and configures the OpenTelemetry layers for different Scuba tables through
/// the following steps:
/// 1. Sets up the global meter provider for metrics using the monarch_metrics table
/// 2. Configures multiple tracing layers with different filters:
///    - Main tracing layer writing to monarch_tracing (excludes specific targets)
///    - Messages layer writing to monarch_messages (only "message" target events)
///    - Executions layer writing to monarch_executions (only "execution" target events)
///
/// Each layer uses its corresponding resource configuration to control:
/// - Which Scuba table to write to
/// - What attributes to include
/// - What data should be filtered/included
pub fn tracing_layer<
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
>() -> impl tracing_subscriber::Layer<S> {
    use tracing_subscriber::prelude::*;
    tracing_subscriber::layer::Identity::new()
        .and_then(
            scuba_tracing::Layer::builder(tracing_resource())
                .build()
                .with_filter(
                    Targets::new()
                        .with_target("hyperactor_telemetry", LevelFilter::OFF)
                        .with_target("message", LevelFilter::OFF)
                        .with_target("execution", LevelFilter::OFF)
                        .with_target("opentelemetry", LevelFilter::OFF)
                        .with_default(LevelFilter::DEBUG),
                ),
        )
        .and_then(
            scuba_tracing::Layer::builder(messages_resource())
                .build()
                .with_filter(Targets::new().with_target("message", LevelFilter::DEBUG)),
        )
        .and_then(
            scuba_tracing::Layer::builder(executions_resource())
                .build()
                .with_filter(Targets::new().with_target("execution", LevelFilter::DEBUG)),
        )
}

/// Creates the base OpenTelemetry resource configuration that all Scuba tables inherit from,
/// configuring common attributes including:
/// - Service name as "monarch/monarch"
/// - Environment variables (job owner, oncall, user, hostname)
/// - Scuba subset based on environment (dev/test/prod)
/// - Execution ID for tracking related events
/// - Column configuration for the execution_id field
fn base_resource() -> Resource {
    let mut builder = opentelemetry_sdk::Resource::builder()
        .with_service_name("monarch/monarch")
        .with_attributes(
            vec![
                pairs_from_env([
                    "MAST_JOB_OWNER_UNIXNAME",
                    "MAST_JOB_OWNER_ONCALL",
                    "USER",
                    "HOSTNAME",
                ]),
                vec![
                    crate::key_value!("execution_id", super::env::execution_id()),
                    crate::key_value!(
                        "meta.scuba.columns",
                        Value::Array(Array::String(vec!["execution_id".into()]))
                    ),
                ],
            ]
            .clone()
            .concat(),
        )
        .with_attributes(
            crate::kv_pairs!(
            "meta.scuba.table" => "monarch_executions",
            "meta.scuba.columns" => Value::Array(Array::String(
                // every row in every table will have these columns
                [
                    "execution_id",
                    "mast_job_owner_unixname",
                    "mast_job_owner_oncall",
                    "user",
                    "hostname",
                ].iter().map(|s| StringValue::from(*s)).collect(),
            )),
            )
            .clone(),
        );

    if let Ok(subset) = std::env::var("SCUBA_SUBSET") {
        builder = builder.with_attribute(key_value!(scuba_utils::SUBSET, subset));
    } else if Env::current() == Env::MastEmulator {
        builder = builder.with_attribute(key_value!(scuba_utils::SUBSET, "mast_emulator"));
    }
    builder.build()
}
/// Configures the OpenTelemetry resource for writing metrics to the "monarch_metrics" Scuba table,
/// inheriting all base attributes and configuring:
/// - Sets meta.scuba.table to "monarch_metrics"
///
/// This table stores all metric data including counters, gauges, and histograms with their attributes
fn metrics_resource() -> Resource {
    Resource::builder()
        .with_attributes(
            base_resource()
                .iter()
                .map(|(key, value)| KeyValue::new(key.clone(), value.clone())),
        )
        .with_attributes(
            kv_pairs!(
            "meta.scuba.table" => "monarch_metrics",
            )
            .clone(),
        )
        .build()
}

/// Configures the OpenTelemetry resource for writing traces to the "monarch_tracing" Scuba table,
/// inheriting all base attributes and configuring:
/// - Sets meta.scuba.table to "monarch_tracing"
///
/// This table stores distributed tracing data including spans, events, and their attributes
fn tracing_resource() -> Resource {
    Resource::builder()
        .with_attributes(
            base_resource()
                .iter()
                .map(|(key, value)| KeyValue::new(key.clone(), value.clone())),
        )
        .with_attributes(
            kv_pairs!(
            "meta.scuba.table" => "monarch_tracing",
            )
            .clone(),
        )
        .build()
}
/// Configures the OpenTelemetry resource for writing messages to the "monarch_messages" Scuba table,
/// inheriting all base attributes and configuring:
/// - Sets meta.scuba.table to "monarch_messages"
///
/// This table specifically stores log messages and events tagged with the "message" target
fn messages_resource() -> Resource {
    Resource::builder()
        .with_attributes(
            base_resource()
                .iter()
                .map(|(key, value)| KeyValue::new(key.clone(), value.clone())),
        )
        .with_attributes(
            kv_pairs!(
            "meta.scuba.table" => "monarch_messages",
            )
            .clone(),
        )
        .build()
}
/// Configures the OpenTelemetry resource for writing execution data to the "monarch_executions" Scuba table,
/// inheriting base attributes (excluding subset) and configuring:
/// - Sets meta.scuba.table to "monarch_executions"
/// - Configures required columns: execution_id, job owner, oncall, user, hostname
///
/// This table tracks high-level execution information and is not split by environment
fn executions_resource() -> Resource {
    Resource::builder()
        .with_attributes(
            base_resource()
                .iter()
                // ignore the subset. This table is not split by env
                .filter(|(k, _v)| **k != scuba_utils::SUBSET)
                .map(|(key, value)| KeyValue::new(key.clone(), value.clone())),
        )
        .with_attributes(
            kv_pairs!(
            "meta.scuba.table" => "monarch_executions",
            )
            .clone(),
        )
        .build()
}
