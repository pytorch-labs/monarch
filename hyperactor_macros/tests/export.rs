/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::Named;
use hyperactor::PortRef;
use hyperactor::data::intern_typename;
use serde::Deserialize;

use crate::Serialize;

#[derive(Debug)]
#[hyperactor::export(TestMessage, (), MyGeneric<()>)]
struct TestActor {
    // Forward the received message to this port, so it can be inspected by
    // the unit test.
    forward_port: PortRef<String>,
}

#[derive(Debug, Clone, Named, Serialize, Deserialize)]
struct TestActorParams {
    forward_port: PortRef<String>,
}

#[async_trait]
impl Actor for TestActor {
    type Params = TestActorParams;

    async fn new(params: Self::Params) -> anyhow::Result<Self> {
        let Self::Params { forward_port } = params;
        Ok(Self { forward_port })
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Named)]
struct TestMessage(String);

#[async_trait]
impl Handler<TestMessage> for TestActor {
    async fn handle(&mut self, this: &Instance<Self>, msg: TestMessage) -> anyhow::Result<()> {
        self.forward_port.send(this, msg.0)?;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct MyGeneric<T>(T);

impl<T: Named> Named for MyGeneric<T> {
    fn typename() -> &'static str {
        intern_typename!(Self, "hyperactor_macros::tests::export::MyGeneric<{}>", T)
    }
}

#[async_trait]
impl Handler<()> for TestActor {
    async fn handle(&mut self, this: &Instance<Self>, _msg: ()) -> anyhow::Result<()> {
        self.forward_port.send(this, "()".to_string())?;
        Ok(())
    }
}

#[async_trait]
impl Handler<MyGeneric<()>> for TestActor {
    async fn handle(&mut self, this: &Instance<Self>, _msg: MyGeneric<()>) -> anyhow::Result<()> {
        self.forward_port.send(this, "MyGeneric<()>".to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use hyperactor::PortRef;
    use hyperactor::proc::Proc;
    use timed_test::async_timed_test;

    use super::*;

    // Ports::new is a private function, so we cannot test it directly. As a
    // workaround, we test whether we can send a message through the message's
    // named port. If the macro is not implemented correctly, the named port
    // will not be bound, and the send will fail.
    #[async_timed_test(timeout_secs = 30)]
    async fn test_binds() {
        let proc = Proc::local();
        let client = proc.attach("client").unwrap();
        let (tx, mut rx) = client.open_port();
        let params = TestActorParams {
            forward_port: tx.bind(),
        };
        let actor_handle = proc.spawn::<TestActor>("foo", params).await.unwrap();
        //  This will call binds
        actor_handle.bind::<TestActor>();
        // Verify that the ports can be gotten successfully.
        {
            // TestMessage type
            let port_id = actor_handle.actor_id().port_id(TestMessage::port());
            let port_ref: PortRef<TestMessage> = PortRef::attest(port_id);
            port_ref
                .send(&client, TestMessage("abc".to_string()))
                .unwrap();
            assert_eq!(rx.recv().await.unwrap(), "abc");
        }
        {
            // () type
            let port_id = actor_handle.actor_id().port_id(<()>::port());
            let port_ref: PortRef<()> = PortRef::attest(port_id);
            port_ref.send(&client, ()).unwrap();
            assert_eq!(rx.recv().await.unwrap(), "()");
        }
        {
            // MyGeneric<()> type
            let port_id = actor_handle.actor_id().port_id(MyGeneric::<()>::port());
            let port_ref: PortRef<MyGeneric<()>> = PortRef::attest(port_id);
            port_ref.send(&client, MyGeneric(())).unwrap();
            assert_eq!(rx.recv().await.unwrap(), "MyGeneric<()>");
        }
    }
}
