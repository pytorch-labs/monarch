from typing import final, Iterator, overload

@final
class SimulatorClient:
    """
    A wrapper around [simulator_client::Simulatorclient] to expose it to python.
    It is a client to communicate with the simulator service.

    Arguments:
    - `proxy_addr`: Address of the simulator's proxy server.
    """

    def __init__(self, proxy_addr: str) -> None: ...
    def kill_world(self, world_name: str) -> None:
        """
        Kill the world with the given name.

        Arguments:
        - `world_name`: Name of the world to kill.
        """
        ...
    def spawn_mesh(
        self, system_addr: str, controller_actor_id: str, worker_world: str
    ) -> None:
        """
        Spawn a mesh actor.

        Arguments:
        - `system_addr`: Address of the system to spawn the mesh in.
        - `controller_actor_id`: Actor id of the controller to spawn the mesh in.
        - `worker_world`: World of the worker to spawn the mesh in.
        """
        ...
