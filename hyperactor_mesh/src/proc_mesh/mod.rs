/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorRef;
use hyperactor::Mailbox;
use hyperactor::RemoteMessage;
use hyperactor::WorldId;
use hyperactor::actor::RemoteActor;
use hyperactor::actor::remote::Remote;
use hyperactor::cap;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::mailbox;
use hyperactor::mailbox::BoxableMailboxSender;
use hyperactor::mailbox::BoxedMailboxSender;
use hyperactor::mailbox::DialMailboxRouter;
use hyperactor::mailbox::MailboxRouter;
use hyperactor::mailbox::MailboxServer;
use hyperactor::mailbox::PortReceiver;
use hyperactor::proc::Proc;
use hyperactor::reference::ProcId;
use hyperactor::reference::Reference;
use hyperactor::supervision::ActorSupervisionEvent;
use ndslice::Range;
use ndslice::Shape;
use ndslice::ShapeError;

use crate::CommActor;
use crate::Mesh;
use crate::actor_mesh::RootActorMesh;
use crate::alloc::Alloc;
use crate::alloc::AllocatorError;
use crate::alloc::ProcState;
use crate::alloc::ProcStopReason;
use crate::assign::Ranks;
use crate::comm::CommActorMode;
use crate::proc_mesh::mesh_agent::MeshAgent;
use crate::proc_mesh::mesh_agent::MeshAgentMessageClient;

pub mod mesh_agent;

use std::sync::OnceLock;

/// A global router shared by all meshes managed in this process;
/// this allows different meshes to communicate with each other.
///
/// This is definitely a "good enough for now" solution; in the future,
/// we'll likely have some form of truly global registration for meshes,
/// also benefitting tooling, etc.
fn global_router() -> &'static MailboxRouter {
    static GLOBAL_ROUTER: OnceLock<MailboxRouter> = OnceLock::new();
    GLOBAL_ROUTER.get_or_init(MailboxRouter::new)
}

/// A ProcMesh maintains a mesh of procs whose lifecycles are managed by
/// an allocator.
pub struct ProcMesh {
    // The underlying set of events. It is None if it has been transferred to
    // a proc event observer.
    event_state: Option<EventState>,
    shape: Shape,
    ranks: Vec<(ProcId, (ChannelAddr, ActorRef<MeshAgent>))>,
    #[allow(dead_code)] // will be used in subsequent diff
    client_proc: Proc,
    client: Mailbox,
    comm_actors: Vec<ActorRef<CommActor>>,
}

struct EventState {
    alloc: Box<dyn Alloc + Send + Sync>,
    supervision_events: PortReceiver<ActorSupervisionEvent>,
}

impl ProcMesh {
    /// Allocate a new ProcMesh from the provided allocator. Allocate returns
    /// after the mesh has been successfully (and fully) allocated, returning
    /// early on any allocation failure.
    pub async fn allocate(
        mut alloc: impl Alloc + Send + Sync + 'static,
    ) -> Result<Self, AllocatorError> {
        // We wait for the full allocation to be running before returning the mesh.
        let shape = alloc.shape().clone();

        let mut proc_ids = Ranks::new(shape.slice().len());
        let mut running = Ranks::new(shape.slice().len());

        while !running.is_full() {
            let Some(state) = alloc.next().await else {
                // Alloc finished before it was fully allocated.
                return Err(AllocatorError::Incomplete(shape));
            };

            match state {
                ProcState::Created { proc_id, coords } => {
                    let rank = shape
                        .slice()
                        .location(&coords)
                        .map_err(|err| AllocatorError::Other(err.into()))?;
                    if let Some(old_proc_id) = proc_ids.insert(rank, proc_id.clone()) {
                        tracing::warn!("rank {rank} reassigned from {old_proc_id} to {proc_id}");
                    }
                    tracing::info!("proc {} rank {}: created", proc_id, rank);
                }
                ProcState::Running {
                    proc_id,
                    mesh_agent,
                    addr,
                } => {
                    let Some(rank) = proc_ids.rank(&proc_id) else {
                        tracing::warn!("proc id {proc_id} running, but not created");
                        continue;
                    };

                    if let Some((old_addr, old_mesh_agent)) =
                        running.insert(*rank, (addr.clone(), mesh_agent.clone()))
                    {
                        tracing::warn!(
                            "duplicate running notifications for {proc_id}, addr:{addr}, mesh_agent:{mesh_agent}, old addr:{old_addr}, old mesh_agent:{old_mesh_agent}"
                        )
                    }
                    tracing::info!(
                        "proc {} rank {}: running at addr:{addr} mesh_agent:{mesh_agent}",
                        proc_id,
                        rank
                    );
                }
                // TODO: We should push responsibility to the allocator, which
                // can choose to either provide a new proc or emit a
                // ProcState::Failed to fail the whole allocation.
                ProcState::Stopped { proc_id, reason } => {
                    tracing::error!("allocation failed for proc_id {}: {}", proc_id, reason);
                    return Err(AllocatorError::Other(anyhow::Error::msg(reason)));
                }
                ProcState::Failed {
                    world_id,
                    description,
                } => {
                    tracing::error!("allocation failed for world {}: {}", world_id, description);
                    return Err(AllocatorError::Other(anyhow::Error::msg(description)));
                }
            }
        }

        // We collect all the ranks at this point of completion, so that we can
        // avoid holding Rcs across awaits.
        let running: Vec<_> = running.into_iter().map(Option::unwrap).collect();

        // All procs are running, so we now configure them.
        let mut world_ids = HashSet::new();

        let (router_channel_addr, router_rx) = channel::serve(ChannelAddr::any(alloc.transport()))
            .await
            .map_err(|err| AllocatorError::Other(err.into()))?;
        let router = DialMailboxRouter::new_with_default(global_router().boxed());
        for (rank, (addr, _agent)) in running.iter().enumerate() {
            let proc_id = proc_ids.get(rank).unwrap().clone();
            router.bind(Reference::Proc(proc_id.clone()), addr.clone());
            // Work around for Allocs that have more than one world.
            world_ids.insert(proc_id.world_id().clone());
        }
        router
            .clone()
            .serve(router_rx, mailbox::monitored_return_handle());

        // Set up a client proc for the mesh itself, so that we can attach ourselves
        // to it, and communicate with the agents. We wire it into the same router as
        // everything else, so now the whole mesh should be able to communicate.
        let client_proc_id = ProcId(WorldId(format!("{}_manager", alloc.world_id().name())), 0);
        let (client_proc_addr, client_rx) = channel::serve(ChannelAddr::any(alloc.transport()))
            .await
            .map_err(|err| AllocatorError::Other(err.into()))?;
        let client_proc = Proc::new(
            client_proc_id.clone(),
            BoxedMailboxSender::new(router.clone()),
        );
        client_proc
            .clone()
            .serve(client_rx, mailbox::monitored_return_handle());
        router.bind(client_proc_id.clone().into(), client_proc_addr);

        // Bind this router to the global router, to enable cross-mesh routing.
        // TODO: unbind this when we incorporate mesh destruction too.
        for world_id in world_ids {
            global_router().bind(world_id.clone().into(), router.clone());
        }
        global_router().bind(alloc.world_id().clone().into(), router.clone());
        global_router().bind(client_proc_id.into(), router.clone());

        let supervisor = client_proc.attach("supervisor")?;
        let (supervison_port, supervision_events) = supervisor.open_port();

        // Now, configure the full mesh, so that the local agents are wired up to
        // our router.
        let client = client_proc.attach("client")?;

        // Map of procs -> channel addresses
        let address_book: HashMap<_, _> = running
            .iter()
            .map(|(addr, agent)| (agent.actor_id().proc_id().clone(), addr.clone()))
            .collect();

        let (config_handle, mut config_receiver) = client.open_port();
        for (rank, (_, agent)) in running.iter().enumerate() {
            agent
                .configure(
                    &client,
                    rank,
                    router_channel_addr.clone(),
                    supervison_port.bind(),
                    address_book.clone(),
                    config_handle.bind(),
                )
                .await?;
        }
        let mut completed = Ranks::new(shape.slice().len());
        while !completed.is_full() {
            let rank = config_receiver
                .recv()
                .await
                .map_err(|err| AllocatorError::Other(err.into()))?;
            if completed.insert(rank, rank).is_some() {
                tracing::warn!("multiple completions received for rank {}", rank);
            }
        }

        // For reasons I fail to fully understand, the below call fails
        // when invoked from `pyo3_async_runtimes::tokio::future_into_py`
        // when using a closure. It appears to be some subtle failure of
        // the compiler to unify lifetimes. If we use a function instead,
        // it does better.
        //
        // Interestingly, this only appears to fail in *specific* caller
        // contexts (e.g., https://fburl.com/code/evfgtfx1), and the error
        // is reported there as "implementation of `std::ops::FnOnce` is not general enough",
        // suggesting some failure of modularity in the compiler's lifetime
        // unification!
        //
        // Baffling and unsettling.
        fn project_actor_ref(pair: &(ChannelAddr, ActorRef<MeshAgent>)) -> ActorRef<MeshAgent> {
            pair.1.clone()
        }

        // Spawn a comm actor on each proc, so that they can be used
        // to perform tree distribution and accumulation.
        let comm_actors = Self::spawn_on_procs::<CommActor>(
            &client,
            running.iter().map(project_actor_ref),
            "comm",
            &Default::default(),
        )
        .await?;
        let address_book: HashMap<_, _> = comm_actors.iter().cloned().enumerate().collect();
        // Now that we have all of the spawned comm actors, kick them all into
        // mesh mode.
        for (rank, comm_actor) in comm_actors.iter().enumerate() {
            comm_actor
                .send(&client, CommActorMode::Mesh(rank, address_book.clone()))
                .map_err(anyhow::Error::from)?;
        }

        let shape = alloc.shape().clone();

        Ok(Self {
            event_state: Some(EventState {
                alloc: Box::new(alloc),
                supervision_events,
            }),
            shape,
            ranks: proc_ids
                .into_iter()
                .map(Option::unwrap)
                .zip(running.into_iter())
                .collect(),
            client_proc,
            client,
            comm_actors,
        })
    }

    async fn spawn_on_procs<A: Actor + RemoteActor>(
        this: &(impl cap::CanSend + cap::CanOpenPort),
        agents: impl IntoIterator<Item = ActorRef<MeshAgent>> + '_,
        actor_name: &str,
        params: &A::Params,
    ) -> Result<Vec<ActorRef<A>>, anyhow::Error>
    where
        A::Params: RemoteMessage,
    {
        let remote = Remote::collect();
        let actor_type = remote
            .name_of::<A>()
            .ok_or(anyhow::anyhow!("actor not registered"))?
            .to_string();

        let (completed_handle, mut completed_receiver) = mailbox::open_port(this);
        let mut n = 0;
        for agent in agents {
            agent
                .gspawn(
                    this,
                    actor_type.clone(),
                    actor_name.to_string(),
                    bincode::serialize(params)?,
                    completed_handle.bind(),
                )
                .await?;
            n += 1;
        }
        let mut completed = Ranks::new(n);
        while !completed.is_full() {
            let (rank, actor_id) = completed_receiver.recv().await?;
            if completed.insert(rank, actor_id).is_some() {
                tracing::warn!("multiple completions received for rank {}", rank);
            }
        }

        // `Ranks` really should have some way to convert into a "completed" rank
        // in a one-shot way; the API here is too awkward otherwise.
        Ok(completed
            .into_iter()
            .map(Option::unwrap)
            .map(ActorRef::attest)
            .collect())
    }

    fn agents(&self) -> impl Iterator<Item = ActorRef<MeshAgent>> + '_ {
        self.ranks.iter().map(|(_, (_, agent))| agent.clone())
    }

    /// Return the comm actor to which casts should be forwarded.
    pub(crate) fn comm_actor(&self) -> &ActorRef<CommActor> {
        &self.comm_actors[0]
    }

    /// Spawn an `ActorMesh` by launching the same actor type on all
    /// agents, using the **same** parameters instance for every
    /// actor.
    ///
    /// - `actor_name`: Name for all spawned actors.
    /// - `params`: Reference to the parameter struct, reused for all
    ///   actors.
    pub async fn spawn<A: Actor + RemoteActor>(
        &self,
        actor_name: &str,
        params: &A::Params,
    ) -> Result<RootActorMesh<'_, A>, anyhow::Error>
    where
        A::Params: RemoteMessage,
    {
        Ok(RootActorMesh::new(
            self,
            actor_name.to_string(),
            Self::spawn_on_procs::<A>(&self.client, self.agents(), actor_name, params).await?,
        ))
    }

    /// A client used to communicate with any member of this mesh.
    pub fn client(&self) -> &Mailbox {
        &self.client
    }

    pub fn client_proc(&self) -> &Proc {
        &self.client_proc
    }

    pub fn proc_id(&self) -> &ProcId {
        self.client_proc.proc_id()
    }

    /// An event stream of proc events. Each ProcMesh can produce only one such
    /// stream, returning None after the first call.
    pub fn events(&mut self) -> Option<ProcEvents> {
        self.event_state.take().map(|event_state| ProcEvents {
            event_state,
            ranks: self
                .ranks
                .iter()
                .enumerate()
                .map(|(rank, (proc_id, _))| (proc_id.clone(), rank))
                .collect(),
        })
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

/// Proc lifecycle events.
#[derive(Debug)]
pub enum ProcEvent {
    /// The proc of the given rank was stopped with the provided reason.
    Stopped(usize, ProcStopReason),
    /// The proc crashed, with the provided "reason". This is reserved for
    /// unhandled supervision events.
    Crashed(usize, String),
}

impl fmt::Display for ProcEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProcEvent::Stopped(rank, reason) => {
                write!(f, "Proc at rank {} stopped: {}", rank, reason)
            }
            ProcEvent::Crashed(rank, reason) => {
                write!(f, "Proc at rank {} crashed: {}", rank, reason)
            }
        }
    }
}

/// An event stream of [`ProcEvent`]
// TODO: consider using streams for this.
pub struct ProcEvents {
    event_state: EventState,
    ranks: HashMap<ProcId, usize>,
}

impl ProcEvents {
    /// Get the next lifecycle event. The stream is closed when this method
    /// returns `None`.
    pub async fn next(&mut self) -> Option<ProcEvent> {
        loop {
            tokio::select! {
                result = self.event_state.alloc.next() => {
                    // Don't disable the outer branch on None: this is always terminal.
                    let Some(alloc_event) = result else {
                        break None;
                    };

                    let ProcState::Stopped { proc_id, reason } = alloc_event else {
                        // Ignore non-stopped events for now.
                        continue;
                    };

                    let Some(rank) = self.ranks.get(&proc_id) else {
                        tracing::warn!("received stop event for unmapped proc {}", proc_id);
                        continue;
                    };

                    break Some(ProcEvent::Stopped(*rank, reason));
                }
                Ok(event) = self.event_state.supervision_events.recv() => {
                    let (actor_id, actor_status) = event.into_inner();
                    let Some(rank) = self.ranks.get(actor_id.proc_id()) else {
                        tracing::warn!("received supervision event for unmapped actor {}", actor_id);
                        continue;
                    };
                    break Some(ProcEvent::Crashed(*rank, actor_status.to_string()))
                }
            }
        }
    }
}

/// Spawns from shared ([`Arc`]) proc meshes, providing [`ActorMesh`]es with
/// static lifetimes.
#[async_trait]
pub trait SharedSpawnable {
    async fn spawn<A: Actor + RemoteActor>(
        &self,
        actor_name: &str,
        params: &A::Params,
    ) -> Result<RootActorMesh<'static, A>, anyhow::Error>
    where
        A::Params: RemoteMessage;
}

#[async_trait]
impl SharedSpawnable for Arc<ProcMesh> {
    async fn spawn<A: Actor + RemoteActor>(
        &self,
        actor_name: &str,
        params: &A::Params,
    ) -> Result<RootActorMesh<'static, A>, anyhow::Error>
    where
        A::Params: RemoteMessage,
    {
        Ok(RootActorMesh::new_shared(
            Arc::clone(self),
            actor_name.to_string(),
            ProcMesh::spawn_on_procs::<A>(&self.client, self.agents(), actor_name, params).await?,
        ))
    }
}

#[async_trait]
impl Mesh for ProcMesh {
    type Node = ProcId;
    type Sliced<'a> = SlicedProcMesh<'a>;

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn select<R: Into<Range>>(
        &self,
        label: &str,
        range: R,
    ) -> Result<Self::Sliced<'_>, ShapeError> {
        Ok(SlicedProcMesh(self, self.shape().select(label, range)?))
    }

    fn get(&self, rank: usize) -> Option<ProcId> {
        Some(self.ranks[rank].0.clone())
    }
}

impl fmt::Display for ProcMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ shape: {} }}", self.shape())
    }
}

impl fmt::Debug for ProcMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProcMesh")
            .field("shape", &self.shape())
            .field("ranks", &self.ranks)
            .field("client_proc", &self.client_proc)
            .field("client", &self.client)
            // Skip the alloc field since it doesn't implement Debug
            .finish()
    }
}

pub struct SlicedProcMesh<'a>(&'a ProcMesh, Shape);

#[async_trait]
impl Mesh for SlicedProcMesh<'_> {
    type Node = ProcId;
    type Sliced<'b>
        = SlicedProcMesh<'b>
    where
        Self: 'b;

    fn shape(&self) -> &Shape {
        &self.1
    }

    fn select<R: Into<Range>>(
        &self,
        label: &str,
        range: R,
    ) -> Result<Self::Sliced<'_>, ShapeError> {
        Ok(Self(self.0, self.1.select(label, range)?))
    }

    fn get(&self, _index: usize) -> Option<ProcId> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use ndslice::shape;

    use super::*;
    use crate::actor_mesh::ActorMesh;
    use crate::actor_mesh::test_util::Error;
    use crate::actor_mesh::test_util::TestActor;
    use crate::alloc::AllocSpec;
    use crate::alloc::Allocator;
    use crate::alloc::local::LocalAllocator;
    use crate::sel_from_shape;

    #[tokio::test]
    async fn test_basic() {
        let alloc = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! { replica = 4 },
                constraints: Default::default(),
            })
            .await
            .unwrap();

        let name = alloc.name().to_string();
        let mesh = ProcMesh::allocate(alloc).await.unwrap();

        assert_eq!(mesh.get(0).unwrap().world_name(), &name);
    }

    #[tokio::test]
    async fn test_propagate_lifecycle_events() {
        let alloc = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! { replica = 4 },
                constraints: Default::default(),
            })
            .await
            .unwrap();

        let stop = alloc.stopper();
        let monkey = alloc.chaos_monkey();
        let mut mesh = ProcMesh::allocate(alloc).await.unwrap();
        let mut events = mesh.events().unwrap();

        monkey(1, ProcStopReason::Killed(1, false));
        assert_matches!(
            events.next().await.unwrap(),
            ProcEvent::Stopped(1, ProcStopReason::Killed(1, false))
        );

        stop();
        for _ in 0..3 {
            assert_matches!(
                events.next().await.unwrap(),
                ProcEvent::Stopped(_, ProcStopReason::Stopped)
            );
        }
        assert!(events.next().await.is_none());
    }

    #[tokio::test]
    async fn test_supervision_failure() {
        // For now, we propagate all actor failures to the proc.

        let alloc = LocalAllocator
            .allocate(AllocSpec {
                shape: shape! { replica = 2  },
                constraints: Default::default(),
            })
            .await
            .unwrap();
        let stop = alloc.stopper();
        let mut mesh = ProcMesh::allocate(alloc).await.unwrap();
        let mut events = mesh.events().unwrap();

        let actors = mesh.spawn::<TestActor>("failing", &()).await.unwrap();

        actors
            .cast(
                sel_from_shape!(actors.shape(), replica = 0),
                Error("failmonkey".to_string()),
            )
            .unwrap();

        assert_matches!(
            events.next().await.unwrap(),
            ProcEvent::Crashed(0, reason) if reason.contains("failmonkey")
        );

        stop();
        assert_matches!(
            events.next().await.unwrap(),
            ProcEvent::Stopped(0, ProcStopReason::Stopped),
        );
        assert_matches!(
            events.next().await.unwrap(),
            ProcEvent::Stopped(1, ProcStopReason::Stopped),
        );

        assert!(events.next().await.is_none());
    }
}
