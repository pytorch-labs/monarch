Device Mesh
===========

Device mesh functionality provides distributed device management and process group operations.
The DeviceMesh class manages collections of devices across multiple processes, enabling
distributed computation patterns.

Classes
-------

.. currentmodule:: monarch.common.device_mesh

.. autosummary::
   :toctree: generated
   :nosignatures:

   DeviceMesh
   RemoteProcessGroup

.. autoclass:: DeviceMesh
   :members:
   :show-inheritance:

.. autoclass:: RemoteProcessGroup
   :members:
   :show-inheritance:

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_active_mesh
   to_mesh
   slice_mesh

.. autofunction:: get_active_mesh

.. autofunction:: to_mesh

.. autofunction:: slice_mesh

The following object is also available from the top-level monarch module:

.. currentmodule:: monarch

.. autodata:: no_mesh
   :annotation: = <_NoMesh object>

   A special mesh context that disables distributed execution.
