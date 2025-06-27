/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::OnceLock;

use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;

use crate::ActorId;
use crate::Message;
use crate::PortId;
use crate::RemoteMessage;
use crate::id;
use crate::mailbox::DeliveryError;
use crate::mailbox::MailboxSender;
use crate::mailbox::MessageEnvelope;
use crate::mailbox::PortHandle;
use crate::mailbox::PortReceiver;
use crate::mailbox::UndeliverableMailboxSender;

/// An undeliverable `M`-typed message (in practice `M` is
/// [MessageEnvelope]).
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Undeliverable<M: Message>(pub M);

/// For `M` a [RemoteMessage], `Undeliverable<M>` is a [Named]
/// instance.
impl<M: RemoteMessage> crate::data::Named for Undeliverable<M> {
    fn typename() -> &'static str {
        crate::data::intern_typename!(Self, "hyperactor::Undeliverable<{}>", M)
    }
}

// Port handle and receiver for undeliverable messages.
pub(crate) fn new_undeliverable_port() -> (
    PortHandle<Undeliverable<MessageEnvelope>>,
    PortReceiver<Undeliverable<MessageEnvelope>>,
) {
    crate::mailbox::Mailbox::new_detached(id!(world[0].proc))
        .open_port::<Undeliverable<MessageEnvelope>>()
}

// An undeliverable message port handle to be shared amongst multiple
// producers. Messages sent here are forwarded to the undeliverable
// mailbox sender.
static MONITORED_RETURN_HANDLE: OnceLock<PortHandle<Undeliverable<MessageEnvelope>>> =
    OnceLock::new();
/// Accessor to the shared monitored undeliverable message port
/// handle. Initialization spawns the undeliverable message port
/// monitor that forwards incoming messages to the undeliverable
/// mailbox sender.
pub fn monitored_return_handle() -> PortHandle<Undeliverable<MessageEnvelope>> {
    let return_handle = MONITORED_RETURN_HANDLE.get_or_init(|| {
        let (return_handle, mut rx) = new_undeliverable_port();
        // Don't reuse `return_handle` for `h`: else it will never get
        // dropped and the task will never return.
        let (h, _) = new_undeliverable_port();
        tokio::spawn(async move {
            while let Ok(Undeliverable(mut envelope)) = rx.recv().await {
                envelope.try_set_error(DeliveryError::BrokenLink(
                    "message returned to undeliverable port".to_string(),
                ));
                super::UndeliverableMailboxSender.post(envelope, /*unused */ h.clone());
            }
        });
        return_handle
    });

    return_handle.clone()
}

/// Returns a message envelope to its original sender.
pub(crate) fn return_undeliverable(
    return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    envelope: MessageEnvelope,
) {
    tokio::spawn(async move {
        let envelope_copy = envelope.clone();
        if (return_handle.send(Undeliverable(envelope))).is_err() {
            UndeliverableMailboxSender.post(envelope_copy, /*unsued*/ return_handle)
        }
    });
}

#[derive(Debug, Error)]
/// Errors that occur during message delivery and return.
pub enum UndeliverableMessageError {
    /// Delivery of a message to its destination failed.
    #[error("a message from {from} to {to} was undeliverable and returned: {error:?}")]
    DeliveryFailure {
        /// The sender of the message.
        from: ActorId,
        /// The destination of the message.
        to: PortId,
        /// Details of why the message couldn't be delivered.
        error: Option<String>,
    },

    /// Delivery of an undeliverable message back to its sender
    /// failed.
    #[error("returning an undeliverable message to sender {sender} failed: {error:?}")]
    ReturnFailure {
        /// The actor the message was to be returned to.
        sender: ActorId,

        /// Details of why the return failed.
        error: Option<String>,
    },
}

impl UndeliverableMessageError {
    /// Constructs `DeliveryFailure` from a failed delivery attempt.
    pub fn delivery_failure(envelope: &MessageEnvelope) -> Self {
        UndeliverableMessageError::DeliveryFailure {
            from: envelope.sender().clone(),
            to: envelope.dest().clone(),
            error: envelope.error().map(|e| format!("{:?}", e)),
        }
    }

    /// Constructs a `ReturnFailure` from a failed return attempt.
    pub fn return_failure(envelope: &MessageEnvelope) -> Self {
        UndeliverableMessageError::ReturnFailure {
            sender: envelope.sender().clone(),
            error: envelope.error().map(|e| format!("{:?}", e)),
        }
    }
}
