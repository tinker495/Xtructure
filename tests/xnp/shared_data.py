"""Shared dataclasses used across xnp tests.

These re-export the canonical test dataclasses from ``tests.testdata`` so the
xnp suite shares one definition with the rest of the component tests.
``HashableData`` is an alias for ``SimpleData``: it is used only as a hashable
carrier in ``unique_mask`` tests, and ``SimpleData`` (same ``id``/``value``
fields) hashes identically.
"""

from tests.testdata import NestedData, SimpleData, VectorData

HashableData = SimpleData

__all__ = ["SimpleData", "VectorData", "NestedData", "HashableData"]
