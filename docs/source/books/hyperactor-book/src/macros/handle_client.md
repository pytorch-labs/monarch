# `#[derive(HandleClient)]`

`#[derive(Handler)]` generates both the server-side handler trait (`ShoppingListHandler`) and the client-side trait definition (`ShoppingListClient`). However, it does not implement the client trait for any specific type.

This is where `#[derive(HandleClient)]` comes in.

## What It Adds

`#[derive(HandleClient)]` generates the following implementation:

```rust
impl<T> ShoppingListClient for ActorHandle<T>
where
  T: ShoppingListHandler + Send + Sync + 'static`
```

This means you can call methods like `.add(...)` or `.list(...)` directly on an `ActorHandle<T>` without needing to manually implement the `ShoppingListClient` trait:

In other words, `HandleClient` connects the generated `ShoppingListClient` interface (from `Handler`) to the concrete type `ActorHandle<T>`.

## Generated Implementation (simplified)
```rust
use async_trait::async_trait;
use hyperactor::{
    ActorHandle,
    anyhow::Error,
    cap::{CanSend, CanOpenPort},
    mailbox::open_once_port,
    metrics,
    Message,
};

#[async_trait]
impl<T> ShoppingListClient for ActorHandle<T>
where
    T: ShoppingListHandler + Send + Sync + 'static,
{
    async fn add(&self, caps: &impl CanSend, item: String) -> Result<(), Error> {
        self.send(caps, ShoppingList::Add(item)).await
    }

    async fn remove(&self, caps: &impl CanSend, item: String) -> Result<(), Error> {
        self.send(caps, ShoppingList::Remove(item)).await
    }

    async fn exists(
        &self,
        caps: &impl CanSend + CanOpenPort,
        item: String,
    ) -> Result<bool, Error> {
        let (reply_to, recv) = open_once_port(caps)?;
        self.send(caps, ShoppingList::Exists(item, reply_to)).await?;
        Ok(recv.await?)
    }

    async fn list(
        &self,
        caps: &impl CanSend + CanOpenPort,
    ) -> Result<Vec<String>, Error> {
        let (reply_to, recv) = open_once_port(caps)?;
        self.send(caps, ShoppingList::List(reply_to)).await?;
        Ok(recv.await?)
    }

```
