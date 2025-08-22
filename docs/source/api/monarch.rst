monarch
=======

.. currentmodule:: monarch

The main Monarch module provides access to all public APIs for distributed computation.

Core Classes
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   DeviceMesh
   RemoteProcessGroup
   Future
   RemoteException
   Shape
   NDSlice
   Selection
   Tensor
   OpaqueRef
   Stream
   Pipe
   ProcessAllocator
   LocalAllocator
   SimAllocator
   ActorFuture
   Simulator

Core Functions
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   remote
   coalescing
   get_active_mesh
   no_mesh
   to_mesh
   slice_mesh
   create_pipe
   remote_generator
   get_active_stream
   reduce
   reduce_
   call_on_shard_and_fetch
   fetch_shard
   inspect
   show
   grad_function
   grad_generator

Mesh Creation Functions
-----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   python_local_mesh
   mast_mesh
   mast_reserve
   rust_backend_mesh
   rust_backend_meshes
   local_mesh
   local_meshes
   rust_mast_mesh
   world_mesh

Utilities
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   timer
   SocketType
   set_meta
   builtins
   function_resolvers