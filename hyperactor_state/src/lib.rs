/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use anyhow::Result;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::Mailbox;
use hyperactor::ProcId;
use hyperactor::actor::Binds;
use hyperactor::actor::RemoteActor;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::id;
use hyperactor::mailbox::BoxedMailboxSender;
use hyperactor::mailbox::DialMailboxRouter;
use hyperactor::mailbox::MailboxClient;
use hyperactor::mailbox::MailboxServer;
use hyperactor::proc::Proc;

pub mod client;
pub mod log_writer;
pub mod object;
pub mod state_actor;

/// Creates a remote client that can send message to actors in the remote addr.
/// It is important to keep the client proc alive for the remote_client's lifetime.
pub async fn create_remote_client(addr: ChannelAddr) -> Result<(Proc, Mailbox)> {
    let remote_sender = MailboxClient::new(channel::dial(addr).unwrap());
    let client_proc_id = id!(client).random_user_proc();
    let client_proc = Proc::new(
        client_proc_id.clone(),
        BoxedMailboxSender::new(remote_sender),
    );
    let remote_client = client_proc.attach("client").unwrap();
    Ok((client_proc, remote_client))
}

pub mod test_utils {
    use super::*;
    use crate::object::GenericStateObject;
    use crate::object::Kind;
    use crate::object::LogSpec;
    use crate::object::LogState;
    use crate::object::Name;
    use crate::object::StateMetadata;
    use crate::object::StateObject;

    pub fn log_items(seq_low: u64, seq_high: u64) -> Vec<GenericStateObject> {
        let mut log_items = vec![];
        let metadata = StateMetadata {
            name: Name::StdoutLog(("test_host".to_string(), 12345)),
            kind: Kind::Log,
        };
        let spec = LogSpec {};
        for seq in seq_low..seq_high {
            let state = LogState::from_string(seq, format!("state {}", seq)).unwrap();
            let state_object =
                StateObject::<LogSpec, LogState>::new(metadata.clone(), spec.clone(), state);
            let generic_state_object = GenericStateObject::try_from(state_object).unwrap();
            log_items.push(generic_state_object);
        }
        log_items
    }

    /// Creates a state actor server at given address. Returns the server address, a handle to the
    /// state actor, and a client mailbox for sending messages to the actor.
    pub async fn spawn_actor<T: Actor + RemoteActor + Binds<T>>(
        addr: ChannelAddr,
        proc_id: ProcId,
        actor_name: &str,
        params: T::Params,
    ) -> Result<(ChannelAddr, ActorHandle<T>, Mailbox)> {
        // Use the provided ProcId directly
        let proc = Proc::new(proc_id, BoxedMailboxSender::new(DialMailboxRouter::new()));

        // Set up the channel server
        let (local_addr, rx) = channel::serve(addr.clone()).await?;

        // Spawn the actor with just a name - the system will generate the full actor ID
        let actor_handle: ActorHandle<T> = proc.spawn(actor_name, params).await?;

        // Create a client mailbox for sending messages to the actor
        let client_mailbox = proc.attach("client")?;

        // Undeliverable messages encountered by the mailbox server
        // are to be returned to the system actor.
        let _mailbox_handle = proc.clone().serve(rx, actor_handle.port());

        // Return the address, handle, and client mailbox
        Ok((local_addr, actor_handle, client_mailbox))
    }
}
