"""Layout-owned conversion policy for adapter reconstruction paths."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def _format_path(path: tuple[str, ...] | str | None) -> str:
    if path is None:
        return "<unknown>"
    if isinstance(path, str):
        return path
    return ".".join(path)


def cast_declared_dtype(
    value: Any,
    declared_dtype: Any,
    *,
    path: tuple[str, ...] | str | None = None,
    context: str,
) -> Any:
    """Cast a reconstructed leaf value to its declared dtype or fail with context.

    Layout Adapters use this helper when an Interface promises declared dtype
    reconstruction.  Silent fallback would make the Interface's error mode
    depend on the caller, so conversion failures are raised with the path and
    context that caused them.
    """
    try:
        return jnp.asarray(value).astype(declared_dtype)
    except TypeError as exc:
        dotted_path = _format_path(path)
        dtype_name = str(jnp.dtype(declared_dtype)) if declared_dtype is not None else "None"
        raise TypeError(
            f"Failed to cast leaf '{dotted_path}' to declared dtype {dtype_name} "
            f"while {context}."
        ) from exc


__all__ = ["cast_declared_dtype"]
