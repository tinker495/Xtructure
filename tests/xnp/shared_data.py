"""Shared dataclasses used across xnp tests."""

import jax.numpy as jnp

from tests.testdata.core import NestedData, SimpleData, VectorData
from xtructure import FieldDescriptor, xtructure_dataclass


@xtructure_dataclass
class HashableData:
    """Dataclass with fields that support hashing for unique_mask tests."""

    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    value: FieldDescriptor.scalar(dtype=jnp.float32)


__all__ = ["SimpleData", "VectorData", "NestedData", "HashableData"]
