use hyperactor_telemetry::declare_static_counter;
use hyperactor_telemetry::declare_static_gauge;
use hyperactor_telemetry::declare_static_histogram;
use hyperactor_telemetry::initialize_logging;

// Declare static metrics for testing
declare_static_counter!(REQUEST_COUNT, "test_requests");
declare_static_gauge!(MEMORY_USAGE, "test_memory_usage");
declare_static_histogram!(REQUEST_DURATION, "test_request_duration");

#[tracing::instrument]
fn something_an_actor_would_do() {
    tracing::debug!("debug message");
}

fn main() {
    // Initialize logging with default configuration
    initialize_logging();
    tracing::info!("info log");
    let _guard = otel_rs::sdk::tracing::FlushGuard {};
}
