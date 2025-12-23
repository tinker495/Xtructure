import jax.numpy as jnp

from xtructure import FieldDescriptor, xtructure_dataclass


@xtructure_dataclass
class PointU32x2:
    x: FieldDescriptor.scalar(dtype=jnp.uint32)
    y: FieldDescriptor.scalar(dtype=jnp.uint32)


@xtructure_dataclass
class HashValueAB:
    a: FieldDescriptor(jnp.uint8)  # type: ignore
    b: FieldDescriptor(jnp.uint32, (1, 2))  # type: ignore


@xtructure_dataclass
class HeapValueABC:
    a: FieldDescriptor(jnp.uint8)  # type: ignore
    b: FieldDescriptor(jnp.uint32, (1, 2))  # type: ignore
    c: FieldDescriptor(jnp.float32, (1, 2, 3))  # type: ignore


@xtructure_dataclass
class OddBytesValue47:
    payload: FieldDescriptor(jnp.uint8, (47,))  # type: ignore
