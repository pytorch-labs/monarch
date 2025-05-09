use std::time::Duration;
use std::time::UNIX_EPOCH;

use anyhow::Result;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::metrics::PeriodicReader;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::metrics::Temporality;
use opentelemetry_sdk::metrics::data::Gauge;
use opentelemetry_sdk::metrics::data::GaugeDataPoint;
use opentelemetry_sdk::metrics::data::Histogram;
use opentelemetry_sdk::metrics::data::HistogramDataPoint;
use opentelemetry_sdk::metrics::data::ResourceMetrics;
use opentelemetry_sdk::metrics::data::Sum;
use opentelemetry_sdk::metrics::data::SumDataPoint;
use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
use scuba::ScubaSample;

use crate::meta::scuba_utils;

struct ScubaExporter {
    scuba: scuba_utils::ScubaClientWrapper,
}

pub struct MeterProviderBuilder {
    resource: opentelemetry_sdk::Resource,
    interval: Duration,
}

impl MeterProviderBuilder {
    pub fn new(resource: opentelemetry_sdk::Resource) -> Self {
        Self {
            resource,
            interval: Duration::from_secs(1),
        }
    }

    pub fn build(self) -> SdkMeterProvider {
        let reader = PeriodicReader::builder(ScubaExporter {
            scuba: scuba_utils::generate_scuba_data(&self.resource),
        })
        .with_interval(self.interval)
        .build();

        SdkMeterProvider::builder()
            .with_resource(self.resource)
            .with_reader(reader)
            .build()
    }
}

trait Numeric: Copy {
    fn into_f64(self) -> f64;
}

impl Numeric for i64 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}
impl Numeric for u64 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}
impl Numeric for f64 {
    fn into_f64(self) -> f64 {
        self
    }
}

impl<T: Numeric> From<&GaugeDataPoint<T>> for MetricSample {
    fn from(value: &GaugeDataPoint<T>) -> Self {
        let mut sample = ScubaSample::new();
        for kv in value.attributes.iter() {
            sample.add(
                kv.key.as_str(),
                scuba::ScubaValue::Normal(kv.value.as_str().to_string()),
            );
        }
        sample.add("value", value.value.into_f64());
        Self(sample)
    }
}

impl<T: Numeric> TryFrom<&HistogramDataPoint<T>> for MetricSample {
    type Error = anyhow::Error;
    fn try_from(value: &HistogramDataPoint<T>) -> Result<Self> {
        let mut sample = ScubaSample::new();
        for kv in value.attributes.iter() {
            sample.add(
                kv.key.as_str(),
                scuba::ScubaValue::Normal(kv.value.as_str().to_string()),
            );
        }

        let mut min = value.min.map(|x| x.into_f64()).unwrap_or_default();
        let mut max = value.max.map(|x| x.into_f64()).unwrap_or_default();
        if min <= 0.0 || max <= min {
            for pct in [0.99, 0.95, 0.90, 0.75, 0.50, 0.25, 0.10] {
                let pct_as_int = (pct * 100.0) as i32;
                let key = format!("p{pct_as_int}");
                sample.add(&key, min);
            }
            sample.add("sum", min);
            sample.add("min", min);
            sample.add("max", min);
            sample.add("avg", min);
            sample.add("stdev", 0.0);
            sample.add("count", value.count as i64);
            return Ok(Self(sample));
        }
        if min < 1.0 {
            min = 1.0;
        }
        if max < min * 2.0 {
            max = min * 2.0;
        }

        let sig = 1000.0;

        // Because hdrhistogram is only based on u64s, and we want 3 sigfigs, multiply then divide by 1k.
        let mut h = hdrhistogram::Histogram::new(4)?;
        for b in value.bounds.iter() {
            for count in value.bucket_counts.iter() {
                if let Err(err) = h.record_n((b * sig) as u64, *count) {
                    tracing::error!("unable to process histogram {err}");
                    return Err(err.into());
                }
            }
        }

        for pct in [0.99, 0.95, 0.90, 0.75, 0.50, 0.25, 0.10] {
            let pct_as_int = (pct * 100.0) as i32;
            let key = format!("p{pct_as_int}");
            sample.add(&key, (h.value_at_percentile(pct) as f64) / sig);
        }
        sample.add("sum", value.sum.into_f64());
        sample.add("min", min);
        sample.add("max", max);
        sample.add("avg", h.mean() / sig);
        sample.add("stdev", h.stdev() / sig);
        sample.add("count", value.count as i64);
        Ok(Self(sample))
    }
}

impl<T: Numeric> From<&Gauge<T>> for SampleSet {
    fn from(value: &Gauge<T>) -> Self {
        Self(
            value
                .data_points
                .iter()
                .map(|point| {
                    let mut sample: MetricSample = point.into();
                    let time = value.time.duration_since(UNIX_EPOCH).unwrap();
                    sample.0.add("time", time.as_secs() as i64);
                    sample
                })
                .collect::<Vec<MetricSample>>(),
        )
    }
}
impl<T: Numeric> From<&SumDataPoint<T>> for MetricSample {
    fn from(value: &SumDataPoint<T>) -> Self {
        let mut sample = ScubaSample::new();
        for kv in value.attributes.iter() {
            sample.add(kv.key.as_str(), kv.value.as_str().to_string());
        }
        sample.add("sum", value.value.into_f64());
        Self(sample)
    }
}

impl<T: Numeric> From<&Sum<T>> for SampleSet {
    fn from(value: &Sum<T>) -> Self {
        let export_interval_ms = value
            .time
            .duration_since(value.start_time)
            .unwrap_or_default()
            .as_millis() as i64;
        Self(
            value
                .data_points
                .iter()
                .map(|point| {
                    let mut sample: MetricSample = point.into();
                    let time = value.time.duration_since(UNIX_EPOCH).unwrap();
                    sample.0.add("time", time.as_secs() as i64);
                    sample.0.add("export_interval_ms", export_interval_ms);
                    sample
                })
                .collect::<Vec<MetricSample>>(),
        )
    }
}

impl<T: Numeric> TryFrom<&Histogram<T>> for SampleSet {
    type Error = anyhow::Error;
    fn try_from(value: &Histogram<T>) -> Result<Self, Self::Error> {
        Ok(Self(
            value
                .data_points
                .iter()
                .map(|point| {
                    let mut sample: MetricSample = point.try_into()?;
                    let time = value.time.duration_since(UNIX_EPOCH).unwrap();
                    sample.0.add("time", time.as_secs() as i64);
                    Ok(sample)
                })
                .collect::<Result<Vec<MetricSample>>>()?,
        ))
    }
}

impl TryFrom<&dyn opentelemetry_sdk::metrics::data::Aggregation> for SampleSet {
    type Error = anyhow::Error;

    fn try_from(
        aggregation: &dyn opentelemetry_sdk::metrics::data::Aggregation,
    ) -> Result<Self, Self::Error> {
        let data = aggregation.as_any();

        // Try to downcast to each type with f64 values
        if let Some(gauge) = data.downcast_ref::<Gauge<f64>>() {
            return Ok(gauge.into());
        } else if let Some(sum) = data.downcast_ref::<Sum<f64>>() {
            return Ok(sum.into());
        } else if let Some(gauge) = data.downcast_ref::<Gauge<u64>>() {
            return Ok(gauge.into());
        } else if let Some(sum) = data.downcast_ref::<Sum<u64>>() {
            return Ok(sum.into());
        } else if let Some(gauge) = data.downcast_ref::<Gauge<i64>>() {
            return Ok(gauge.into());
        } else if let Some(sum) = data.downcast_ref::<Sum<i64>>() {
            return Ok(sum.into());
        } else if let Some(h) = data.downcast_ref::<Histogram<u64>>() {
            return h.try_into();
        } else if let Some(h) = data.downcast_ref::<Histogram<i64>>() {
            return h.try_into();
        } else if let Some(h) = data.downcast_ref::<Histogram<f64>>() {
            return h.try_into();
        }

        Err(anyhow::anyhow!(
            "Unknown aggregation type: {:?}",
            data.type_id()
        ))
    }
}

struct MetricSample(ScubaSample);

struct SampleSet(Vec<MetricSample>);

impl ScubaExporter {
    pub fn builder(resource: opentelemetry_sdk::Resource) -> MeterProviderBuilder {
        MeterProviderBuilder::new(resource)
    }
}

impl PushMetricExporter for ScubaExporter {
    async fn export(&self, metrics: &mut ResourceMetrics) -> OTelSdkResult {
        for scope in &metrics.scope_metrics {
            for metric in &scope.metrics {
                match SampleSet::try_from(metric.data.as_ref()) {
                    Ok(samples) => {
                        for mut sample in samples.0 {
                            sample.0.add("key", metric.name.to_string());
                            sample.0.add("unit", metric.unit.to_string());
                            sample.0.add("scope", scope.scope.name());
                            self.scuba.log(sample.0);
                        }
                    }
                    Err(err) => {
                        tracing::error!(
                            "error reached when encoding metric key: {} {}",
                            metric.name,
                            err,
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Flushes any metric data held by an exporter.
    fn force_flush(&self) -> OTelSdkResult {
        self.scuba.flush();
        Ok(())
    }

    /// Releases any held computational resources.
    ///
    /// After Shutdown is called, calls to Export will perform no operation and
    /// instead will return an error indicating the shutdown state.
    fn shutdown(&self) -> OTelSdkResult {
        self.force_flush()
    }

    /// Access the [Temporality] of the MetricExporter.
    fn temporality(&self) -> Temporality {
        Temporality::Delta
    }
}
