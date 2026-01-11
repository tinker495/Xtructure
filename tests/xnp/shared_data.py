"""Shared dataclasses used across xnp tests."""

import jax.numpy as jnp

from xtructure import FieldDescriptor, xtructure_dataclass


@xtructure_dataclass
class SimpleData:
    """Basic scalar dataclass used in multiple xnp tests."""

    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    value: FieldDescriptor.scalar(dtype=jnp.float32)


@xtructure_dataclass
class VectorData:
    """Dataclass with fixed vector fields for batch and vector ops."""

    position: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))
    velocity: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


@xtructure_dataclass
class NestedData:
    """Nested dataclass combining simple and vector fields."""

    simple: FieldDescriptor.scalar(dtype=SimpleData)
    vector: FieldDescriptor.scalar(dtype=VectorData)


@xtructure_dataclass
class HashableData:
    """Dataclass with fields that support hashing for unique_mask tests."""

    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    value: FieldDescriptor.scalar(dtype=jnp.float32)
