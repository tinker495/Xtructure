import jax.numpy as jnp

from xtructure import FieldDescriptor, xtructure_dataclass


@xtructure_dataclass
class SimpleData:
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    value: FieldDescriptor.scalar(dtype=jnp.float32)


@xtructure_dataclass
class VectorData:
    position: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))
    velocity: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


@xtructure_dataclass
class NestedData:
    simple: FieldDescriptor.scalar(dtype=SimpleData)
    vector: FieldDescriptor.scalar(dtype=VectorData)
