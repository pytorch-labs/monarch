/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::LazyLock;

use hyperactor::ActorId;
use hyperactor::Instance;
use hyperactor::Mailbox;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::id;
use hyperactor::mailbox::AliasingMailboxSender;
use hyperactor::mailbox::MailboxClient;
use hyperactor::mailbox::MailboxServer;
use tokio::sync::OnceCell;

use crate::proc_mesh::global_root_client;
use crate::proc_mesh::global_router;

pub static EXTERNAL_DEBUG_ROUTER_ID: LazyLock<ActorId> =
    LazyLock::new(|| id!(debug_router[0].debug_router[0]));

/// Used in the process being debugged.
///
/// Start a mailbox server at a known address that external
/// debugging processes will be able to send messages to. Additionally,
/// in the global router, bind EXTERNAL_DEBUG_ROUTER_ID so that any
/// messages sent to it are posted to the actual debug controller
/// singleton's mailbox. The AliasingMailboxSender will intrusively
/// replace the message envelope's destination actor id with the
/// true debug controller actor's id so that the mailbox will accept
/// the message. EXTERNAL_DEBUG_ROUTER_ID is used because the debug
/// controller's actor id isn't known until runtime and there isn't
/// a good way for the external debugging process to discover it.
pub async fn init_debug_server(
    debug_controller_mailbox: Mailbox,
    listen_addr: ChannelAddr,
) -> anyhow::Result<()> {
    static DEBUG_SERVER_INIT: OnceCell<()> = OnceCell::const_new();
    DEBUG_SERVER_INIT
        .get_or_try_init(async || -> anyhow::Result<()> {
            let (_, debug_rx) = channel::serve(listen_addr).await?;
            global_router().clone().serve(debug_rx);
            global_router().bind(
                EXTERNAL_DEBUG_ROUTER_ID.clone().into(),
                AliasingMailboxSender::new(
                    EXTERNAL_DEBUG_ROUTER_ID.clone(),
                    debug_controller_mailbox,
                ),
            );
            Ok(())
        })
        .await?;
    Ok(())
}

/// Used in the external debugger process.
///
/// Returns the global root client, but configures it so that:
/// 1) It begins listening for responses on listen_addr.
/// 2) Any messages sent to EXTERNAL_DEBUG_ROUTER_ID are
///    forwarded to server_addr (which must be the same
///    as the listen_addr passed to init_debug_server).
pub async fn debug_cli_client(
    server_addr: ChannelAddr,
    listen_addr: ChannelAddr,
) -> anyhow::Result<&'static Instance<()>> {
    static DEBUG_CLI_CLIENT: OnceCell<&'static Instance<()>> = OnceCell::const_new();
    Ok(DEBUG_CLI_CLIENT
        .get_or_try_init(async || -> anyhow::Result<&'static Instance<()>> {
            let (_, rx) = channel::serve(listen_addr).await?;
            MailboxServer::serve(global_root_client(), rx);
            global_router().bind(
                EXTERNAL_DEBUG_ROUTER_ID.clone().into(),
                MailboxClient::new(channel::dial(server_addr)?),
            );
            Ok(global_root_client())
        })
        .await?)
}

/// Used in the process being debugged.
///
/// When the debug controller receives a message from an external debugger
/// process (identified by debug_cli_actor_id) for the first time, it needs
/// to bind debug_cli_actor_id to response_addr in the global router
/// in order to be able to deliver responses back to the external process.
/// response_addr must be the same as listen_addr in debug_cli_client().
pub fn bind_debug_cli_actor(
    debug_cli_actor_id: ActorId,
    response_addr: ChannelAddr,
) -> anyhow::Result<()> {
    Ok(global_router().bind(
        debug_cli_actor_id.into(),
        MailboxClient::new(channel::dial(response_addr)?),
    ))
}
