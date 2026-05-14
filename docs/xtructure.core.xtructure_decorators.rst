xtructure.core.xtructure\_decorators package
============================================

Module contents
---------------

.. automodule:: xtructure.core.xtructure_decorators
   :members:
   :show-inheritance:
   :undoc-members:

Adapter module layout
---------------------

Decorator implementation modules are organized by adapter role:

- ``layout_adapters`` owns layout-derived behavior such as defaults, shape
  properties, indexing, structure utilities, validation, bitpack accessors, and
  aggregate bitpack views.
- ``pytree_adapters`` owns PyTree-level behavior such as hashing, IO method
  attachment, comparison operators, and string formatting.

Compatibility import aliases
----------------------------

The pre-reorganization module paths under
``xtructure.core.xtructure_decorators`` remain import-compatible aliases for
the corresponding ``layout_adapters`` and ``pytree_adapters`` modules.
