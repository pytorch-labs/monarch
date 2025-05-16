use hyperactor_telemetry::*;

declare_static_timer!(
    ACTOR_MESH_CAST_DURATION,
    "actor_mesh_cast_duration",
    TimeUnit::Micros
);
