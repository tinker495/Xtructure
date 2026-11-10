import jax.numpy as jnp

from xtructure import FieldDescriptor, Xtructurable


class PointU32x2(Xtructurable):
    x: FieldDescriptor.scalar(dtype=jnp.uint32)
    y: FieldDescriptor.scalar(dtype=jnp.uint32)


class HashValueAB(Xtructurable):
    a: FieldDescriptor(jnp.uint8)  # type: ignore
    b: FieldDescriptor(jnp.uint32, (1, 2))  # type: ignore


class HeapValueABC(Xtructurable):
    a: FieldDescriptor(jnp.uint8)  # type: ignore
    b: FieldDescriptor(jnp.uint32, (1, 2))  # type: ignore
    c: FieldDescriptor(jnp.float32, (1, 2, 3))  # type: ignore


class OddBytesValue47(Xtructurable):
    payload: FieldDescriptor(jnp.uint8, (47,))  # type: ignore
