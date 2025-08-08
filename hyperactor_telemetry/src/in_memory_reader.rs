/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::sync::Weak;

use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::metrics::InstrumentKind;
use opentelemetry_sdk::metrics::ManualReader;
use opentelemetry_sdk::metrics::MetricResult;
use opentelemetry_sdk::metrics::Pipeline;
use opentelemetry_sdk::metrics::Temporality;
use opentelemetry_sdk::metrics::data::ResourceMetrics;
use opentelemetry_sdk::metrics::data::Sum;
use opentelemetry_sdk::metrics::reader::MetricReader;

// Global ManualReader instance for easy access with cumulative temporality
static IN_MEMORY_MANUAL_READER: std::sync::LazyLock<ManualReader> =
    std::sync::LazyLock::new(|| {
        ManualReader::builder()
            .with_temporality(Temporality::Cumulative)
            .build()
    });

/// InMemoryReader that wraps the global ManualReader and implements MetricReader
#[derive(Debug)]
pub struct InMemoryReader;

impl InMemoryReader {
    pub fn new() -> Self {
        Self
    }
}

impl MetricReader for InMemoryReader {
    fn register_pipeline(&self, pipeline: Weak<Pipeline>) {
        IN_MEMORY_MANUAL_READER.register_pipeline(pipeline);
    }

    fn collect(&self, rm: &mut ResourceMetrics) -> MetricResult<()> {
        IN_MEMORY_MANUAL_READER.collect(rm)
    }

    fn force_flush(&self) -> OTelSdkResult {
        IN_MEMORY_MANUAL_READER.force_flush()
    }

    fn shutdown(&self) -> OTelSdkResult {
        IN_MEMORY_MANUAL_READER.shutdown()
    }

    fn temporality(&self, kind: InstrumentKind) -> Temporality {
        IN_MEMORY_MANUAL_READER.temporality(kind)
    }
}

// Public API for In Memory Metrics
impl InMemoryReader {
    /// Get all counters from the global ManualReader
    pub fn get_all_counters(&self) -> HashMap<String, i64> {
        let mut rm = ResourceMetrics {
            resource: Resource::builder_empty().build(),
            scope_metrics: Vec::new(),
        };
        let _ = IN_MEMORY_MANUAL_READER.collect(&mut rm);

        // Extract counters directly from the collected metrics
        let mut counters = HashMap::new();
        for scope in &rm.scope_metrics {
            for metric in &scope.metrics {
                let data = metric.data.as_any();

                if let Some(sum_u64) = data.downcast_ref::<Sum<u64>>() {
                    for data_point in &sum_u64.data_points {
                        let metric_name = metric.name.to_string();
                        counters.insert(metric_name, data_point.value as i64);
                    }
                } else if let Some(sum_i64) = data.downcast_ref::<Sum<i64>>() {
                    for data_point in &sum_i64.data_points {
                        let metric_name = metric.name.to_string();
                        counters.insert(metric_name, data_point.value);
                    }
                }
            }
        }
        counters
    }
}

#[cfg(test)]
mod tests {
    use opentelemetry_sdk::metrics::SdkMeterProvider;

    use super::*;

    #[test]
    fn test_get_all_counters() {
        let provider = SdkMeterProvider::builder()
            .with_reader(InMemoryReader::new())
            .build();

        opentelemetry::global::set_meter_provider(provider);

        // Create static counters using the macro
        crate::declare_static_counter!(TEST_COUNTER_1, "test_counter_1");
        crate::declare_static_counter!(TEST_COUNTER_2, "test_counter_2");

        // Bump the counters with different values
        TEST_COUNTER_1.add(10, &[]);
        TEST_COUNTER_2.add(25, &[]);
        TEST_COUNTER_1.add(5, &[]); // Add more to the first counter (total should be 15)

        // Get all counters and verify values
        let counters = InMemoryReader::new().get_all_counters();

        // The counters should contain our test counters
        println!("All counters: {:?}", counters);

        // Assert that we have counters
        assert!(!counters.is_empty(), "Should have some counters");

        // Assert specific counter values
        // TEST_COUNTER_1 should have 15 (10 + 5)
        // TEST_COUNTER_2 should have 25
        assert_eq!(
            counters.get("test_counter_1"),
            Some(&15),
            "TEST_COUNTER_1 should be 15"
        );
        assert_eq!(
            counters.get("test_counter_2"),
            Some(&25),
            "TEST_COUNTER_2 should be 25"
        );
    }

    #[test]
    fn test_get_all_counters_empty() {
        // Get counters when none have been created
        let counters = InMemoryReader::new().get_all_counters();

        // Should be empty
        println!("Empty counters: {:?}", counters);

        // This test ensures the function doesn't panic when no counters exist
        assert!(
            counters.is_empty(),
            "Should be empty when no counters created"
        );
    }
}
