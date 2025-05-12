pub fn tracing_layer<
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
>() -> Option<impl tracing_subscriber::Layer<S>> {
    #[cfg(fbcode_build)]
    {
        Some(crate::meta::tracing_layer())
    }
    #[cfg(not(fbcode_build))]
    {
        None::<Box<dyn tracing_subscriber::Layer<S> + Send + Sync>>
    }
}

pub fn init_metrics() {
    #[cfg(fbcode_build)]
    {
        opentelemetry::global::set_meter_provider(crate::meta::meter_provider());
    }
}
