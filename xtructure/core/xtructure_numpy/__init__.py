from __future__ import annotations

from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp

from xtructure.core.type_utils import is_xtructure_dataclass_instance

from . import dataclass_ops as _dc
from .array_ops import _update_array_on_condition


def _is_xtructurable(value: Any) -> bool:
    return is_xtructure_dataclass_instance(value)


def _coerce_sequence(values: Iterable[Any]) -> list[Any]:
    return list(values)


def _check_homogeneous_inputs(func_name: str, arrays_list: list[Any]) -> bool:
    if not arrays_list:
        raise ValueError(f"Cannot {func_name} empty list.")
    is_dataclass = [_is_xtructurable(arr) for arr in arrays_list]
    if any(is_dataclass) and not all(is_dataclass):
        raise TypeError(f"{func_name} inputs must be all xtructure dataclasses or all arrays.")
    return all(is_dataclass)


def _reject_dataclass_kwargs(func_name: str, **kwargs: Any) -> None:
    rejected = {name: value for name, value in kwargs.items() if value is not None}
    if rejected:
        keys = ", ".join(sorted(rejected.keys()))
        raise TypeError(f"{func_name} does not support {keys} for xtructure dataclass inputs.")


def concat(arrays, /, *, axis: int | None = 0):
    arrays_list = _coerce_sequence(arrays)
    if _check_homogeneous_inputs("concat", arrays_list):
        if axis is None:
            return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=None), *arrays_list)
        return _dc.concat(arrays_list, axis=axis)
    return jnp.concat(arrays_list, axis=axis)


def concatenate(arrays, axis: int | None = 0, dtype: Any | None = None):
    arrays_list = _coerce_sequence(arrays)
    if _check_homogeneous_inputs("concatenate", arrays_list):
        if dtype is not None:
            raise TypeError("concatenate does not support dtype for xtructure dataclass inputs.")
        if axis is None:
            return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=None), *arrays_list)
        return _dc.concat(arrays_list, axis=axis)
    return jnp.concatenate(arrays_list, axis=axis, dtype=dtype)


def pad(array, pad_width, mode: str | Any = "constant", **kwargs):
    if _is_xtructurable(array):
        return _dc.pad(array, pad_width, mode=mode, **kwargs)
    return jnp.pad(array, pad_width, mode=mode, **kwargs)


def stack(arrays, axis: int = 0, out: None = None, dtype: Any | None = None):
    arrays_list = _coerce_sequence(arrays)
    if _check_homogeneous_inputs("stack", arrays_list):
        _reject_dataclass_kwargs("stack", out=out, dtype=dtype)
        return _dc.stack(arrays_list, axis=axis)
    return jnp.stack(arrays_list, axis=axis, out=out, dtype=dtype)


def reshape(a, shape, order: str = "C", *, copy: bool | None = None, out_sharding=None):
    if _is_xtructurable(a):
        if order != "C":
            raise ValueError("xtructure reshape only supports order='C'.")
        _reject_dataclass_kwargs("reshape", copy=copy, out_sharding=out_sharding)
        shape_tuple = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        return _dc.reshape(a, shape_tuple)
    return jnp.reshape(a, shape, order=order, copy=copy, out_sharding=out_sharding)


def ravel(a, order: str = "C", *, out_sharding=None):
    if _is_xtructurable(a):
        if order != "C":
            raise ValueError("xtructure ravel only supports order='C'.")
        _reject_dataclass_kwargs("ravel", out_sharding=out_sharding)
        return _dc.flatten(a)
    return jnp.ravel(a, order=order, out_sharding=out_sharding)


def flatten(array: Any, order: str = "C") -> Any:
    return ravel(array, order=order)


def where(condition, x=None, y=None, /, *, size=None, fill_value=None):
    if x is None and y is None:
        return jnp.where(condition, x, y, size=size, fill_value=fill_value)
    if _is_xtructurable(x) or _is_xtructurable(y):
        if x is None or y is None:
            raise TypeError("x and y must be provided for xtructure dataclass inputs.")
        if _is_xtructurable(y) and not _is_xtructurable(x):
            raise TypeError("x and y must both be xtructure dataclasses for dataclass where.")
        if _is_xtructurable(x) and _is_xtructurable(y):
            if jax.tree_util.tree_structure(x) != jax.tree_util.tree_structure(y):
                raise TypeError("x and y must have the same tree structure.")
        _reject_dataclass_kwargs("where", size=size, fill_value=fill_value)
        return _dc.where(condition, x, y)
    return jnp.where(condition, x, y, size=size, fill_value=fill_value)


def where_no_broadcast(condition: Any, x: Any, y: Any) -> Any:
    return _dc.where_no_broadcast(condition, x, y)


def take(
    a,
    indices,
    axis: int | None = None,
    out=None,
    mode: str | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value=None,
):
    if _is_xtructurable(a):
        _reject_dataclass_kwargs("take", out=out)
        return jax.tree_util.tree_map(
            lambda x: jnp.take(
                x,
                indices,
                axis=axis,
                mode=mode,
                unique_indices=unique_indices,
                indices_are_sorted=indices_are_sorted,
                fill_value=fill_value,
            ),
            a,
        )
    return jnp.take(
        a,
        indices,
        axis=axis,
        out=out,
        mode=mode,
        unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted,
        fill_value=fill_value,
    )


def take_along_axis(arr, indices, axis: int | None = -1, mode=None, fill_value=None):
    if _is_xtructurable(arr):
        if mode is not None or fill_value is not None:
            return jax.tree_util.tree_map(
                lambda x: jnp.take_along_axis(
                    x, indices, axis=axis, mode=mode, fill_value=fill_value
                ),
                arr,
            )
        return _dc.take_along_axis(arr, indices, axis=axis)
    return jnp.take_along_axis(arr, indices, axis=axis, mode=mode, fill_value=fill_value)


def tile(A, reps):
    if _is_xtructurable(A):
        return _dc.tile(A, reps)
    return jnp.tile(A, reps)


def transpose(a, axes: Sequence[int] | None = None):
    if _is_xtructurable(a):
        return _dc.transpose(a, axes=axes)
    return jnp.transpose(a, axes=axes)


def swapaxes(a, axis1: int, axis2: int):
    if _is_xtructurable(a):
        return _dc.swapaxes(a, axis1=axis1, axis2=axis2)
    return jnp.swapaxes(a, axis1, axis2)


def unique_mask(
    val: Any,
    key: Any | None = None,
    filled: Any | None = None,
    key_fn: Any | None = None,
    batch_len: int | None = None,
    return_index: bool = False,
    return_inverse: bool = False,
) -> Any:
    return _dc.unique_mask(
        val,
        key=key,
        filled=filled,
        key_fn=key_fn,
        batch_len=batch_len,
        return_index=return_index,
        return_inverse=return_inverse,
    )


def update_on_condition(dataclass_instance, indices, condition, values_to_set):
    if _is_xtructurable(dataclass_instance):
        if _is_xtructurable(values_to_set):
            if jax.tree_util.tree_structure(values_to_set) != jax.tree_util.tree_structure(
                dataclass_instance
            ):
                raise TypeError(
                    "values_to_set must have the same tree structure as dataclass_instance."
                )
        return _dc.update_on_condition(dataclass_instance, indices, condition, values_to_set)
    if _is_xtructurable(values_to_set):
        raise TypeError("values_to_set must not be an xtructure dataclass when updating an array.")
    return _update_array_on_condition(dataclass_instance, indices, condition, values_to_set)


def expand_dims(a, axis: int | Sequence[int]):
    if _is_xtructurable(a):
        return _dc.expand_dims(a, axis=axis)
    return jnp.expand_dims(a, axis=axis)


def squeeze(a, axis: int | Sequence[int] | None = None):
    if _is_xtructurable(a):
        return _dc.squeeze(a, axis=axis)
    return jnp.squeeze(a, axis=axis)


def repeat(
    a,
    repeats,
    axis: int | None = None,
    *,
    total_repeat_length: int | None = None,
    out_sharding=None,
):
    if _is_xtructurable(a):
        _reject_dataclass_kwargs(
            "repeat", total_repeat_length=total_repeat_length, out_sharding=out_sharding
        )
        return _dc.repeat(a, repeats, axis=axis)
    return jnp.repeat(
        a,
        repeats,
        axis=axis,
        total_repeat_length=total_repeat_length,
        out_sharding=out_sharding,
    )


def split(ary, indices_or_sections, axis: int = 0):
    if _is_xtructurable(ary):
        return _dc.split(ary, indices_or_sections, axis=axis)
    return list(jnp.split(ary, indices_or_sections, axis=axis))


def full_like(a, fill_value, dtype: Any | None = None, shape: Any = None, *, device=None):
    if _is_xtructurable(a):
        _reject_dataclass_kwargs("full_like", dtype=dtype, shape=shape, device=device)
        return _dc.full_like(a, fill_value)
    return jnp.full_like(a, fill_value, dtype=dtype, shape=shape, device=device)


def zeros_like(a, dtype: Any | None = None, shape: Any = None, *, device=None):
    if _is_xtructurable(a):
        _reject_dataclass_kwargs("zeros_like", dtype=dtype, shape=shape, device=device)
        return _dc.zeros_like(a)
    return jnp.zeros_like(a, dtype=dtype, shape=shape, device=device)


def ones_like(a, dtype: Any | None = None, shape: Any = None, *, device=None):
    if _is_xtructurable(a):
        _reject_dataclass_kwargs("ones_like", dtype=dtype, shape=shape, device=device)
        return _dc.ones_like(a)
    return jnp.ones_like(a, dtype=dtype, shape=shape, device=device)


__all__ = [
    "concat",
    "concatenate",
    "pad",
    "stack",
    "reshape",
    "ravel",
    "flatten",
    "where",
    "where_no_broadcast",
    "take",
    "take_along_axis",
    "tile",
    "transpose",
    "swapaxes",
    "unique_mask",
    "update_on_condition",
    "expand_dims",
    "squeeze",
    "repeat",
    "split",
    "zeros_like",
    "ones_like",
    "full_like",
]
