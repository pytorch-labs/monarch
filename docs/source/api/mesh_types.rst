Mesh Types
==========

Mesh types provide different backend implementations for distributed computation.
These functions create and manage various types of computation meshes for different deployment scenarios.

Local Mesh Functions
--------------------

.. currentmodule:: monarch.python_local_mesh

.. autosummary::
   :toctree: generated
   :nosignatures:

   python_local_mesh

.. autofunction:: python_local_mesh

.. currentmodule:: monarch.rust_local_mesh

.. autosummary::
   :toctree: generated
   :nosignatures:

   local_mesh
   local_meshes

.. autofunction:: local_mesh

.. autofunction:: local_meshes

Rust Backend Functions
----------------------

.. currentmodule:: monarch.rust_backend_mesh

.. autosummary::
   :toctree: generated
   :nosignatures:

   rust_backend_mesh
   rust_backend_meshes
   rust_mast_mesh

.. autofunction:: rust_backend_mesh

.. autofunction:: rust_backend_meshes

.. autofunction:: rust_mast_mesh

World Mesh Functions
--------------------

.. currentmodule:: monarch.world_mesh

.. autosummary::
   :toctree: generated
   :nosignatures:

   world_mesh

.. autofunction:: world_mesh

Notebook Integration
-------------------

.. currentmodule:: monarch.notebook

.. autosummary::
   :toctree: generated
   :nosignatures:

   mast_mesh
   reserve_torchx

.. autofunction:: mast_mesh

.. autofunction:: reserve_torchx

Classes
-------

.. currentmodule:: monarch.rust_local_mesh

.. autosummary::
   :toctree: generated
   :nosignatures:

   SocketType

.. autoclass:: SocketType
   :members:
   :show-inheritance:
