import jax.numpy as jnp

from xtructure import FieldDescriptor, Xtructurable


class SimpleData(Xtructurable):
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    value: FieldDescriptor.scalar(dtype=jnp.float32)


class VectorData(Xtructurable):
    position: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))
    velocity: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


class NestedData(Xtructurable):
    simple: FieldDescriptor.scalar(dtype=SimpleData)
    vector: FieldDescriptor.scalar(dtype=VectorData)
