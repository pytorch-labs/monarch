/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;
use std::time::Duration;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::Context;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Named;
use hyperactor::OncePortRef;
use hyperactor::RefClient;
use hyperactor::proc::Proc;
use serde::Deserialize;
use serde::Serialize;

#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
enum ShoppingList {
    // Oneway messages dispatch messages asynchronously, with no reply.
    Add(String),
    Remove(String),

    // Call messages dispatch a request, expecting a reply to the
    // provided port, which must be in the last position.
    Exists(String, #[reply] OncePortRef<bool>),

    List(#[reply] OncePortRef<Vec<String>>),
}

// Define an actor.
#[derive(Debug)]
#[hyperactor::export(
    spawn = true,
    handlers = [
        ShoppingList,
    ],
)]
struct ShoppingListActor(HashSet<String>);

#[async_trait]
impl Actor for ShoppingListActor {
    type Params = ();

    async fn new(_params: ()) -> Result<Self, anyhow::Error> {
        Ok(Self(HashSet::new()))
    }
}

// ShoppingListHandler is the trait generated by derive(Handler) above.
// We implement the trait here for the actor, defining a handler for
// each ShoppingList message.
//
// The `forward` attribute installs a handler that forwards messages
// to the `ShoppingListHandler` implementation directly. This can also
// be done manually:
//
// ```ignore
//<ShoppingListActor as ShoppingListHandler>
//     ::handle(self, comm, message).await
// ```
#[async_trait]
#[hyperactor::forward(ShoppingList)]
impl ShoppingListHandler for ShoppingListActor {
    async fn add(&mut self, _cx: &Context<Self>, item: String) -> Result<(), anyhow::Error> {
        eprintln!("insert {}", item);
        self.0.insert(item);
        Ok(())
    }

    async fn remove(&mut self, _cx: &Context<Self>, item: String) -> Result<(), anyhow::Error> {
        eprintln!("remove {}", item);
        self.0.remove(&item);
        Ok(())
    }

    async fn exists(&mut self, _cx: &Context<Self>, item: String) -> Result<bool, anyhow::Error> {
        Ok(self.0.contains(&item))
    }

    async fn list(&mut self, _cx: &Context<Self>) -> Result<Vec<String>, anyhow::Error> {
        Ok(self.0.iter().cloned().collect())
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let mut proc = Proc::local();

    // Spawn our actor, and get a handle for rank 0.
    let shopping_list_actor: hyperactor::ActorHandle<ShoppingListActor> =
        proc.spawn("shopping", ()).await?;

    // We join the system, so that we can send messages to actors.
    let client = proc.attach("client").unwrap();

    // todo: consider making this a macro to remove the magic names

    // Derive(Handler) generates client methods, which call the
    // remote handler provided an instance (send + open capability),
    // the destination actor, and the method arguments.

    shopping_list_actor.add(&client, "milk".into()).await?;
    shopping_list_actor.add(&client, "eggs".into()).await?;

    println!(
        "got milk? {}",
        shopping_list_actor.exists(&client, "milk".into()).await?
    );
    println!(
        "got yoghurt? {}",
        shopping_list_actor
            .exists(&client, "yoghurt".into())
            .await?
    );

    shopping_list_actor.remove(&client, "milk".into()).await?;
    println!(
        "got milk now? {}",
        shopping_list_actor.exists(&client, "milk".into()).await?
    );

    println!(
        "shopping list: {:?}",
        shopping_list_actor.list(&client).await?
    );

    let _ = proc.destroy_and_wait(Duration::from_secs(1), None).await?;
    Ok(())
}
