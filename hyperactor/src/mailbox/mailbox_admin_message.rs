use serde::Deserialize;
use serde::Serialize;

pub use crate as hyperactor;
use crate::HandleClient;
use crate::Handler;
use crate::Named;
use crate::ProcId;
use crate::RefClient;
use crate::mailbox::ChannelAddr;

/// Messages relating to mailbox administration.
#[derive(
    Handler,
    HandleClient,
    RefClient,
    Debug,
    Serialize,
    Deserialize,
    Clone,
    PartialEq,
    Named
)]
pub enum MailboxAdminMessage {
    /// An address update.
    UpdateAddress {
        /// The ID of the proc.
        proc_id: ProcId,

        /// The address at which it listens.
        addr: ChannelAddr,
    },
}
