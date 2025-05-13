//! An implementation of mocked controller that can be used for simulation.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor::Named;
use hyperactor::actor::ActorHandle;
use hyperactor::channel::ChannelAddr;
use hyperactor::data::Serialized;
use hyperactor_multiprocess::proc_actor::ProcActor;
use hyperactor_multiprocess::proc_actor::spawn;
use hyperactor_multiprocess::system_actor::ProcLifecycleMode;
use monarch_messages::client::ClientActor;
use monarch_messages::client::ClientMessageClient;
use monarch_messages::controller::WorkerError;
use monarch_messages::controller::*;
use monarch_messages::debugger::DebuggerAction;
use monarch_messages::worker::Ref;
use monarch_messages::worker::WorkerMessage;
use serde::Deserialize;
use serde::Serialize;
use tokio::sync::OnceCell;

use crate::worker::WorkerActor;

#[derive(Debug)]
#[hyperactor::export_spawn(ControllerMessage)]
pub struct SimControllerActor {
    client_actor_ref: OnceCell<ActorRef<ClientActor>>,
    worker_actor_ref: ActorRef<WorkerActor>,
    /// A light weight map from seq to result.
    history: HashMap<Seq, Result<Serialized, WorkerError>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub struct SimControllerParams {
    worker_actor_id: ActorId,
}

impl SimControllerParams {
    pub fn new(worker_actor_id: ActorId) -> Self {
        // Only a single worker is created here to simulate a gang of workers.
        Self { worker_actor_id }
    }
}

#[async_trait]
impl Actor for SimControllerActor {
    type Params = SimControllerParams;

    async fn new(params: SimControllerParams) -> Result<Self, anyhow::Error> {
        Ok(Self {
            client_actor_ref: OnceCell::new(),
            worker_actor_ref: ActorRef::attest(params.worker_actor_id),
            history: HashMap::new(),
        })
    }

    async fn init(&mut self, _this: &hyperactor::Instance<Self>) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

#[async_trait]
#[hyperactor::forward(ControllerMessage)]
impl ControllerMessageHandler for SimControllerActor {
    async fn attach(
        &mut self,
        _this: &hyperactor::Instance<Self>,
        client_actor: ActorRef<ClientActor>,
    ) -> Result<(), anyhow::Error> {
        self.client_actor_ref
            .set(client_actor)
            .map_err(|actor_ref| anyhow::anyhow!("client actor {} already attached", actor_ref))
    }

    async fn node(
        &mut self,
        _this: &hyperactor::Instance<Self>,
        _seq: Seq,
        _uses: Vec<Ref>,
        _defs: Vec<Ref>,
    ) -> Result<(), anyhow::Error> {
        tracing::info!("controller node: uses: {:?}, defs: {:?}", _uses, _defs);
        Ok(())
    }

    async fn send(
        &mut self,
        this: &hyperactor::Instance<Self>,
        ranks: Ranks,
        message: Serialized,
    ) -> Result<(), anyhow::Error> {
        tracing::info!("controller send to ranks {:?}: {}", ranks, message);
        self.worker_actor_ref
            .port::<WorkerMessage>()
            .send_serialized(this, &message);
        Ok(())
    }

    async fn remote_function_failed(
        &mut self,
        _this: &hyperactor::Instance<Self>,
        _seq: Seq,
        _error: WorkerError,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }

    async fn status(
        &mut self,
        this: &hyperactor::Instance<Self>,
        seq: Seq,
        _worker_actor_id: ActorId,
        controller: bool,
    ) -> Result<(), anyhow::Error> {
        tracing::info!(
            "controller status: seq {}, worker_actor_id {:?}, controller: {:?}",
            &seq,
            _worker_actor_id,
            controller
        );
        tracing::info!("controller history in status(): {:?}", &self.history);
        let result = self.history.remove(&seq).unwrap();
        let client = self.client_actor_ref.get().unwrap();
        if let Err(e) = client
            .result(this, seq.clone(), Some(Ok(result.unwrap())))
            .await
        {
            tracing::error!("controller failed to send result: {:?}", e);
        }
        Ok(())
    }

    async fn fetch_result(
        &mut self,
        _this: &hyperactor::Instance<Self>,
        seq: Seq,
        result: Result<Serialized, WorkerError>,
    ) -> Result<(), anyhow::Error> {
        tracing::info!(
            "controller fetch result: seq {}, result {}",
            &seq,
            &result.as_ref().unwrap()
        );
        self.history.insert(seq, result);
        tracing::info!("controller history in fetch_result: {:?}", &self.history);
        Ok(())
    }

    async fn check_supervision(
        &mut self,
        _this: &hyperactor::Instance<Self>,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }

    async fn drop_refs(
        &mut self,
        _this: &hyperactor::Instance<Self>,
        _refs: Vec<Ref>,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }

    async fn debugger_message(
        &mut self,
        _this: &hyperactor::Instance<Self>,
        _debugger_actor_id: ActorId,
        _action: DebuggerAction,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }

    #[cfg(test)]
    async fn get_first_incomplete_seqs_unit_tests_only(
        &mut self,
        _this: &hyperactor::Instance<Self>,
    ) -> Result<Vec<Seq>, anyhow::Error> {
        Ok(vec![])
    }

    #[cfg(not(test))]
    async fn get_first_incomplete_seqs_unit_tests_only(
        &mut self,
        _this: &hyperactor::Instance<Self>,
    ) -> Result<Vec<Seq>, anyhow::Error> {
        unimplemented!("get_first_incomplete_seqs_unit_tests_only is only for unit tests")
    }
}

impl SimControllerActor {
    /// Bootstrap the controller actor. This will create a new proc, join the system at `bootstrap_addr`
    /// and spawn the controller actor into the proc.
    pub async fn bootstrap(
        controller_id: ActorId,
        listen_addr: ChannelAddr,
        bootstrap_addr: ChannelAddr,
        params: SimControllerParams,
        supervision_update_interval: Duration,
    ) -> Result<(ActorHandle<ProcActor>, ActorRef<SimControllerActor>), anyhow::Error> {
        let bootstrap = ProcActor::bootstrap(
            controller_id.proc_id().clone(),
            controller_id.proc_id().world_id().clone(), // REFACTOR(marius): plumb world id through SimControllerActor::bootstrap
            listen_addr,
            bootstrap_addr.clone(),
            supervision_update_interval,
            HashMap::new(),
            ProcLifecycleMode::ManagedBySystem,
        )
        .await?;

        let mut system = hyperactor_multiprocess::System::new(bootstrap_addr);
        let client = system.attach().await?;

        let controller_actor_ref = spawn::<SimControllerActor>(
            &client,
            &bootstrap.proc_actor.bind(),
            controller_id.clone().name(),
            &params,
        )
        .await?;

        Ok((bootstrap.proc_actor, controller_actor_ref))
    }

    #[allow(dead_code)]
    fn client(&self) -> Result<ActorRef<ClientActor>, anyhow::Error> {
        self.client_actor_ref
            .get()
            .ok_or_else(|| anyhow::anyhow!("client actor ref not set"))
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use hyperactor::channel::ChannelTransport;
    use hyperactor::id;
    use hyperactor_multiprocess::System;
    use hyperactor_multiprocess::system_actor::SystemActorParams;

    use super::*;

    #[tokio::test]
    async fn test_bootstrap() {
        let server_handle = System::serve(
            ChannelAddr::any(ChannelTransport::Local),
            SystemActorParams::new(Duration::from_secs(10), Duration::from_secs(10)),
        )
        .await
        .unwrap();

        let controller_id = id!(controller[0].root);
        let (proc_handle, actor_ref) = SimControllerActor::bootstrap(
            controller_id.clone(),
            ChannelAddr::any(ChannelTransport::Local),
            server_handle.local_addr().clone(),
            SimControllerParams {
                worker_actor_id: id!(worker[0].root),
            },
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        assert_eq!(*actor_ref.actor_id(), controller_id);

        proc_handle.drain_and_stop().unwrap();
    }
}
