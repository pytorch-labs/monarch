use anyhow::anyhow;
use hyperactor::PortId;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::Tx;
use hyperactor::channel::dial;
use hyperactor::data::Serialized;
use hyperactor::id;
use hyperactor::mailbox::MessageEnvelope;
use hyperactor::simnet::OperationalMessage;
use hyperactor::simnet::ProxyMessage;
use hyperactor::simnet::SpawnMesh;
use monarch_hyperactor::runtime::get_tokio_runtime;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tokio::sync::oneshot;

/// A wrapper around [ndslice::Slice] to expose it to python.
/// It is a compact representation of indices into the flat
/// representation of an n-dimensional array. Given an offset, sizes of
/// each dimension, and strides for each dimension, Slice can compute
/// indices into the flat array.
#[pyclass(
    name = "SimulatorClient",
    frozen,
    module = "monarch._monarch.simulator_client"
)]
#[derive(Clone)]
pub(crate) struct SimulatorClient {
    proxy_addr: ChannelAddr,
}

fn wrap_operational_message(operational_message: OperationalMessage) -> MessageEnvelope {
    let serialized_operational_message = Serialized::serialize(&operational_message).unwrap();
    let proxy_message = ProxyMessage::new(None, serialized_operational_message);
    let serialized_proxy_message = Serialized::serialize(&proxy_message).unwrap();
    let sender_id = id!(simulator_client[0].sender_actor);
    // a dummy port ID. We are delivering message with low level mailbox.
    // The port ID is not used.
    let port_id = PortId(id!(simulator[0].actor), 0);
    MessageEnvelope::new(sender_id, port_id, serialized_proxy_message)
}

#[pymethods]
impl SimulatorClient {
    #[new]
    fn new(proxy_addr: &str) -> PyResult<Self> {
        Ok(Self {
            proxy_addr: proxy_addr
                .parse::<ChannelAddr>()
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
        })
    }

    fn kill_world(&self, world_name: &str) -> PyResult<()> {
        let operational_message = OperationalMessage::KillWorld(world_name.to_string());
        let external_message = wrap_operational_message(operational_message);
        get_tokio_runtime()
            .block_on(async {
                let tx = dial(self.proxy_addr.clone()).map_err(|err| anyhow!(err))?;
                tx.post(external_message, oneshot::channel().0)
                    .await
                    .map_err(|err| anyhow!("Failed to post message: {}", err))
            })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(())
    }

    fn spawn_mesh(
        &self,
        system_addr: &str,
        controller_actor_id: &str,
        worker_world: &str,
    ) -> PyResult<()> {
        let spawn_mesh = SpawnMesh::new(
            system_addr.parse().unwrap(),
            controller_actor_id.parse().unwrap(),
            worker_world.parse().unwrap(),
        );
        let operational_message = OperationalMessage::SpawnMesh(spawn_mesh);
        let external_message = wrap_operational_message(operational_message);
        get_tokio_runtime()
            .block_on(async {
                let tx = dial(self.proxy_addr.clone()).map_err(|err| anyhow!(err))?;
                tx.post(external_message, oneshot::channel().0)
                    .await
                    .map_err(|err| anyhow!("Failed to post message: {}", err))
            })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(())
    }
}

pub(crate) fn init_pymodule(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let simulator_client_mod = PyModule::new_bound(module.py(), "simulator_client")?;
    simulator_client_mod.add_class::<SimulatorClient>()?;
    module.add_submodule(&simulator_client_mod)?;
    Ok(())
}
