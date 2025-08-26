Python API
==========

.. note::
   This documents Monarch's public APIs.


.. contents:: Quick Navigation
   :local:
   :depth: 2

Core Components
===============

Core building blocks for distributed computing: meshes, streams, futures, and data structures.

.. autofunction:: monarch.future.ActorFuture

.. autofunction:: monarch.common.device_mesh.DeviceMesh

.. autofunction:: monarch.common.future.Future

.. autofunction:: monarch._src.actor.shape.NDSlice

.. autofunction:: monarch.common.pipe.Pipe

.. autofunction:: monarch.common.device_mesh.RemoteProcessGroup

.. autofunction:: monarch.common.selection.Selection

.. autofunction:: monarch._src.actor.shape.Shape

.. autofunction:: monarch.common.stream.Stream

.. autofunction:: monarch.common.pipe.create_pipe

.. autofunction:: monarch.common.device_mesh.get_active_mesh

.. autofunction:: monarch.common.stream.get_active_stream

.. autofunction:: monarch.common.device_mesh.no_mesh

.. autofunction:: monarch.common.pipe.remote_generator

.. autofunction:: monarch.common.device_mesh.slice_mesh

.. autofunction:: monarch.common.device_mesh.to_mesh

Mesh Creation & Management
==========================

Functions for creating and managing different types of compute meshes.

.. autofunction:: monarch.rust_local_mesh.SocketType

.. autofunction:: monarch.rust_local_mesh.local_mesh

.. autofunction:: monarch.rust_local_mesh.local_meshes

.. autofunction:: monarch.notebook.mast_mesh

.. autofunction:: monarch.notebook.reserve_torchx

.. autofunction:: monarch.python_local_mesh.python_local_mesh

.. autofunction:: monarch.rust_backend_mesh.rust_backend_mesh

.. autofunction:: monarch.rust_backend_mesh.rust_backend_meshes

.. autofunction:: monarch.rust_backend_mesh.rust_mast_mesh

.. autofunction:: monarch.world_mesh.world_mesh

Data Operations
===============

Operations for data processing, fetching, and gradient computation.

.. autofunction:: monarch.common.tensor.Tensor

.. autofunction:: monarch.fetch.call_on_shard_and_fetch

.. autofunction:: monarch.fetch.fetch_shard

.. autofunction:: monarch.gradient_generator.grad_function

.. autofunction:: monarch.gradient_generator.grad_generator

.. autofunction:: monarch.fetch.inspect

.. autofunction:: monarch.common.tensor.reduce

.. autofunction:: monarch.common.tensor.reduce_

.. autofunction:: monarch.fetch.show

Remote Execution
================

Remote function execution and distributed computing primitives.

.. autofunction:: monarch.common.invocation.RemoteException

.. autofunction:: monarch.common.remote.remote

System & Utilities
==================

System utilities, allocators, configuration, and helper functions.

.. autofunction:: monarch._src.actor.allocator.LocalAllocator

.. autofunction:: monarch.common.opaque_ref.OpaqueRef

.. autofunction:: monarch._src.actor.allocator.ProcessAllocator

.. autofunction:: monarch._src_actor.allocator.SimAllocator

.. autofunction:: monarch.simulator.interface.Simulator

.. autofunction:: monarch.builtins.builtins

.. autofunction:: monarch.common._coalescing.coalescing

.. autofunction:: monarch.common.function.resolvers

.. autofunction:: monarch.simulator.config.set_meta

.. autofunction:: monarch.timer.timer

Alphabetical Index
==================

All APIs in alphabetical order:

* :py:func:`~monarch.ActorFuture`
* :py:func:`~monarch.DeviceMesh`
* :py:func:`~monarch.Future`
* :py:func:`~monarch.LocalAllocator`
* :py:func:`~monarch.NDSlice`
* :py:func:`~monarch.OpaqueRef`
* :py:func:`~monarch.Pipe`
* :py:func:`~monarch.ProcessAllocator`
* :py:func:`~monarch.RemoteException`
* :py:func:`~monarch.RemoteProcessGroup`
* :py:func:`~monarch.Selection`
* :py:func:`~monarch.Shape`
* :py:func:`~monarch.SimAllocator`
* :py:func:`~monarch.Simulator`
* :py:func:`~monarch.SocketType`
* :py:func:`~monarch.Stream`
* :py:func:`~monarch.Tensor`
* :py:func:`~monarch.builtins`
* :py:func:`~monarch.call_on_shard_and_fetch`
* :py:func:`~monarch.coalescing`
* :py:func:`~monarch.create_pipe`
* :py:func:`~monarch.fetch_shard`
* :py:func:`~monarch.function_resolvers`
* :py:func:`~monarch.get_active_mesh`
* :py:func:`~monarch.get_active_stream`
* :py:func:`~monarch.grad_function`
* :py:func:`~monarch.grad_generator`
* :py:func:`~monarch.inspect`
* :py:func:`~monarch.local_mesh`
* :py:func:`~monarch.local_meshes`
* :py:func:`~monarch.mast_mesh`
* :py:func:`~monarch.mast_reserve`
* :py:func:`~monarch.no_mesh`
* :py:func:`~monarch.python_local_mesh`
* :py:func:`~monarch.reduce`
* :py:func:`~monarch.reduce_`
* :py:func:`~monarch.remote`
* :py:func:`~monarch.remote_generator`
* :py:func:`~monarch.rust_backend_mesh`
* :py:func:`~monarch.rust_backend_meshes`
* :py:func:`~monarch.rust_mast_mesh`
* :py:func:`~monarch.set_meta`
* :py:func:`~monarch.show`
* :py:func:`~monarch.slice_mesh`
* :py:func:`~monarch.timer`
* :py:func:`~monarch.to_mesh`
* :py:func:`~monarch.world_mesh`
