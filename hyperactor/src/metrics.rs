//! A bunch of statily defined metrics. Defined here because they are used in
//! both macros and handwritten code.

use hyperactor_telemetry::declare_static_counter;
use hyperactor_telemetry::declare_static_timer;
use hyperactor_telemetry::declare_static_up_down_counter;

declare_static_counter!(MESSAGES_SENT, "messages_sent");
declare_static_counter!(MESSAGES_RECEIVED, "messages_received");
declare_static_counter!(MESSAGE_HANDLE_ERRORS, "message_handle_errors");
declare_static_counter!(MESSAGE_RECEIVE_ERRORS, "message_receive_errors");
declare_static_up_down_counter!(MESSAGE_QUEUE_SIZE, "message_queue_size");
declare_static_timer!(
    MESSAGE_HANDLER_DURATION,
    "message_handler_duration",
    hyperactor_telemetry::TimeUnit::Nanos
);

declare_static_timer!(
    ACTOR_STATUS,
    "actor.status",
    hyperactor_telemetry::TimeUnit::Nanos
);
