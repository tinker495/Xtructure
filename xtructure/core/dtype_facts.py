"""Internal DType Kind facts for primitive JAX dtypes.

The **DType Kind** axis is part of the Internal Layout Adapter Interface.
It classifies primitive JAX dtypes once, then Layout Adapters consume the
named fact instead of re-implementing bool / unsigned / signed / floating
switches locally.
"""

from __future__ import annotations

import enum
from typing import Any

import jax.numpy as jnp

SIZE_DTYPE = jnp.uint32


class DTypeKind(enum.Enum):
    """Recognised primitive **DType Kind** values."""

    BOOL = "bool"
    UINT = "uint"
    INT = "int"
    FLOAT = "float"


def dtype_kind(dtype: Any) -> DTypeKind:
    """Return the **DType Kind** for a primitive JAX dtype.

    Dtypes outside bool, unsigned integer, signed integer, and floating point
    are rejected. Nested xtructure fields use the orthogonal ``field_kind``
    axis and must not enter this primitive classifier.
    """

    try:
        dtype_obj = jnp.dtype(dtype)
    except TypeError as exc:
        raise TypeError(f"Unsupported DType Kind for dtype {dtype!r}.") from exc

    if jnp.issubdtype(dtype_obj, jnp.bool_):
        return DTypeKind.BOOL
    if jnp.issubdtype(dtype_obj, jnp.unsignedinteger):
        return DTypeKind.UINT
    if jnp.issubdtype(dtype_obj, jnp.integer):
        return DTypeKind.INT
    if jnp.issubdtype(dtype_obj, jnp.floating):
        return DTypeKind.FLOAT
    raise TypeError(
        "Unsupported DType Kind for dtype "
        f"{dtype!r}: expected bool, unsigned integer, signed integer, or floating point."
    )


def is_supported_primitive_dtype(dtype: Any) -> bool:
    """Return True when ``dtype`` has a recognised **DType Kind**."""

    try:
        dtype_kind(dtype)
    except TypeError:
        return False
    return True


def default_fill_value_for_dtype(dtype: Any) -> Any:
    """Return the default fill value for a primitive dtype via **DType Kind**."""

    kind = dtype_kind(dtype)
    if kind is DTypeKind.BOOL:
        return False
    if kind is DTypeKind.UINT:
        return jnp.iinfo(jnp.dtype(dtype)).max
    if kind is DTypeKind.INT:
        return 0
    if kind is DTypeKind.FLOAT:
        return jnp.inf
    raise AssertionError(f"Unhandled DType Kind: {kind!r}")


def unsigned_integer_dtype_for(dtype: Any) -> Any:
    """Return the same-width unsigned dtype for a signed integer dtype."""

    kind = dtype_kind(dtype)
    if kind is not DTypeKind.INT:
        raise TypeError(f"Expected signed integer DType Kind, got {kind.value!r}.")
    bits = jnp.iinfo(jnp.dtype(dtype)).bits
    return jnp.dtype(f"uint{bits}")
