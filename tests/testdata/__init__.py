"""Shared test-only xtructure dataclasses.

Centralizing test dataclasses makes it easier to:
- reuse the same value types across multiple component tests (HashTable/BGPQ/Queue/Stack/IO)
- run generic tests over multiple dataclass types via parametrization
"""

from .core import NestedData, SimpleData, VectorData
from .values import HashValueAB, HeapValueABC, OddBytesValue47, PointU32x2

__all__ = [
    "SimpleData",
    "VectorData",
    "NestedData",
    "PointU32x2",
    "HashValueAB",
    "HeapValueABC",
    "OddBytesValue47",
]
