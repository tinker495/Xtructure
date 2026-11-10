from __future__ import annotations

import inspect
from typing import Any, Iterable, Sequence, overload

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

from xtructure.core.protocol import Xtructurable, XtructurableT

from . import dataclass_ops as _dc
from .array_ops import _update_array_on_condition


def _check_homogeneous_inputs(func_name: str, arrays_list: list[Any]) -> bool:
    if not arrays_list:
        raise ValueError(f"Cannot {func_name} empty list.")
    is_dataclass = [isinstance(arr, Xtructurable) for arr in arrays_list]
    if any(is_dataclass) and not all(is_dataclass):
        raise TypeError(f"{func_name} inputs must be all xtructure dataclasses or all arrays.")
    return all(is_dataclass)


def _reject_dataclass_kwargs(func_name: str, **kwargs: Any) -> None:
    rejected = {name: value for name, value in kwargs.items() if value is not None}
    if rejected:
        keys = ", ".join(sorted(rejected.keys()))
        raise TypeError(f"{func_name} does not support {keys} for xtructure dataclass inputs.")


def _supports_keyword(func: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        param.name == name or param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters
    )


def _like_kwargs(
    func_name: str,
    jnp_func: Any,
    *,
    dtype: Any | None,
    shape: Any,
    device: Any,
    out_sharding: Any,
) -> dict[str, Any]:
    kwargs = {"dtype": dtype, "shape": shape, "device": device}
    if out_sharding is None:
        return kwargs
    if not _supports_keyword(jnp_func, "out_sharding"):
        raise TypeError(
            f"{func_name} out_sharding requires a JAX version whose jnp.{func_name} "
            "supports out_sharding."
        )
    kwargs["out_sharding"] = out_sharding
    return kwargs


@overload
def concat(arrays: Iterable[XtructurableT], /, *, axis: int | None = 0) -> XtructurableT:
    ...


@overload
def concat(arrays: Iterable[ArrayLike], /, *, axis: int | None = 0) -> jax.Array:
    ...


def concat(arrays, /, *, axis: int | None = 0):
    arrays_list = list(arrays)
    if _check_homogeneous_inputs("concat", arrays_list):
        if axis is None:
            return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=None), *arrays_list)
        return _dc.concat(arrays_list, axis=axis)
    return jnp.concat(arrays_list, axis=axis)


@overload
def concatenate(
    arrays: Iterable[XtructurableT], axis: int | None = 0, dtype: None = None
) -> XtructurableT:
    ...


@overload
def concatenate(
    arrays: Iterable[ArrayLike], axis: int | None = 0, dtype: DTypeLike | None = None
) -> jax.Array:
    ...


def concatenate(arrays, axis: int | None = 0, dtype: Any | None = None) -> Xtructurable | jax.Array:
    arrays_list = list(arrays)
    if _check_homogeneous_inputs("concatenate", arrays_list):
        if dtype is not None:
            raise TypeError("concatenate does not support dtype for xtructure dataclass inputs.")
        if axis is None:
            return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=None), *arrays_list)
        return _dc.concat(arrays_list, axis=axis)
    return jnp.concatenate(arrays_list, axis=axis, dtype=dtype)


@overload
def pad(
    array: XtructurableT, pad_width: Any, mode: str | Any = "constant", **kwargs: Any
) -> XtructurableT:
    ...


@overload
def pad(array: ArrayLike, pad_width: Any, mode: str | Any = "constant", **kwargs: Any) -> jax.Array:
    ...


def pad(array, pad_width, mode: str | Any = "constant", **kwargs):
    if isinstance(array, Xtructurable):
        return _dc.pad(array, pad_width, mode=mode, **kwargs)
    return jnp.pad(array, pad_width, mode=mode, **kwargs)


@overload
def stack(
    arrays: Iterable[XtructurableT], axis: int = 0, out: None = None, dtype: None = None
) -> XtructurableT:
    ...


@overload
def stack(
    arrays: Iterable[ArrayLike], axis: int = 0, out: None = None, dtype: Any | None = None
) -> jax.Array:
    ...


def stack(arrays, axis: int = 0, out: None = None, dtype: Any | None = None):
    arrays_list = list(arrays)
    if _check_homogeneous_inputs("stack", arrays_list):
        _reject_dataclass_kwargs("stack", out=out, dtype=dtype)
        return _dc.stack(arrays_list, axis=axis)
    return jnp.stack(arrays_list, axis=axis, out=out, dtype=dtype)


@overload
def reshape(
    a: XtructurableT, shape: Any, order: str = "C", *, copy: None = None, out_sharding: None = None
) -> XtructurableT:
    ...


@overload
def reshape(
    a: ArrayLike,
    shape: Any,
    order: str = "C",
    *,
    copy: bool | None = None,
    out_sharding: Any = None,
) -> jax.Array:
    ...


def reshape(a, shape, order: str = "C", *, copy: bool | None = None, out_sharding=None):
    if isinstance(a, Xtructurable):
        if order != "C":
            raise ValueError("xtructure reshape only supports order='C'.")
        _reject_dataclass_kwargs("reshape", copy=copy, out_sharding=out_sharding)
        shape_tuple = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        return a.reshape(shape_tuple)
    return jnp.reshape(a, shape, order=order, copy=copy, out_sharding=out_sharding)


@overload
def ravel(a: XtructurableT, order: str = "C", *, out_sharding: None = None) -> XtructurableT:
    ...


@overload
def ravel(a: ArrayLike, order: str = "C", *, out_sharding: Any = None) -> jax.Array:
    ...


def ravel(a, order: str = "C", *, out_sharding=None):
    if isinstance(a, Xtructurable):
        if order != "C":
            raise ValueError("xtructure ravel only supports order='C'.")
        _reject_dataclass_kwargs("ravel", out_sharding=out_sharding)
        return a.flatten()
    return jnp.ravel(a, order=order, out_sharding=out_sharding)


@overload
def flatten(array: XtructurableT, order: str = "C") -> XtructurableT:
    ...


@overload
def flatten(array: ArrayLike, order: str = "C") -> jax.Array:
    ...


def flatten(array, order: str = "C"):
    return ravel(array, order=order)


@overload
def where(
    condition: ArrayLike,
    x: XtructurableT,
    y: XtructurableT,
    /,
    *,
    size: None = None,
    fill_value: None = None,
) -> XtructurableT:
    ...


@overload
def where(
    condition: ArrayLike,
    x: XtructurableT,
    y: ArrayLike,
    /,
    *,
    size: None = None,
    fill_value: None = None,
) -> XtructurableT:
    ...


@overload
def where(
    condition: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    /,
    *,
    size: None = None,
    fill_value: None = None,
) -> jax.Array:
    ...


@overload
def where(
    condition: ArrayLike,
    x: None = None,
    y: None = None,
    /,
    *,
    size: int | None = None,
    fill_value: ArrayLike | tuple[ArrayLike, ...] | None = None,
) -> tuple[jax.Array, ...]:
    ...


def where(
    condition: ArrayLike,
    x: XtructurableT | ArrayLike | None = None,
    y: XtructurableT | ArrayLike | None = None,
    /,
    *,
    size: int | None = None,
    fill_value: ArrayLike | tuple[ArrayLike, ...] | None = None,
) -> XtructurableT | jax.Array | tuple[jax.Array, ...]:
    if x is None and y is None:
        return jnp.where(condition, size=size, fill_value=fill_value)
    if isinstance(x, Xtructurable):
        if y is None:
            raise TypeError("x and y must be provided for xtructure dataclass inputs.")
        if isinstance(y, Xtructurable):
            if jax.tree_util.tree_structure(x) != jax.tree_util.tree_structure(y):
                raise TypeError("x and y must have the same tree structure.")
        _reject_dataclass_kwargs("where", size=size, fill_value=fill_value)
        return _dc.where(condition, x, y)
    if isinstance(y, Xtructurable):
        if x is None:
            raise TypeError("x and y must be provided for xtructure dataclass inputs.")
        raise TypeError("x and y must both be xtructure dataclasses for dataclass where.")
    if x is None or y is None:
        raise TypeError("x and y must be provided together for array inputs.")
    if size is not None or fill_value is not None:
        raise ValueError(
            "size and fill_value arguments cannot be used in three-term where function."
        )
    return jnp.where(condition, x, y)


def where_no_broadcast(condition: Any, x: Any, y: Any) -> Any:
    return _dc.where_no_broadcast(condition, x, y)


@overload
def take(
    a: XtructurableT,
    indices: ArrayLike,
    axis: int | None = None,
    out: None = None,
    mode: str | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value=None,
) -> XtructurableT:
    ...


@overload
def take(
    a: ArrayLike,
    indices: ArrayLike,
    axis: int | None = None,
    out: None = None,
    mode: str | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value=None,
) -> jax.Array:
    ...


def take(
    a: XtructurableT | ArrayLike,
    indices: ArrayLike,
    axis: int | None = None,
    out=None,
    mode: str | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value=None,
) -> XtructurableT | jax.Array:
    if isinstance(a, Xtructurable):
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


@overload
def take_along_axis(
    arr: XtructurableT,
    indices: ArrayLike,
    axis: int | None = -1,
    mode: Any = None,
    fill_value: Any = None,
    *,
    wrap_negative_indices: bool = True,
) -> XtructurableT:
    ...


@overload
def take_along_axis(
    arr: ArrayLike,
    indices: ArrayLike,
    axis: int | None = -1,
    mode: Any = None,
    fill_value: Any = None,
    *,
    wrap_negative_indices: bool = True,
) -> jax.Array:
    ...


def take_along_axis(
    arr,
    indices,
    axis: int | None = -1,
    mode=None,
    fill_value=None,
    *,
    wrap_negative_indices: bool = True,
):
    kwargs = dict(mode=mode, fill_value=fill_value, wrap_negative_indices=wrap_negative_indices)
    if isinstance(arr, Xtructurable):
        return _dc.take_along_axis(arr, indices, axis=axis, **kwargs)
    return jnp.take_along_axis(arr, indices, axis=axis, **kwargs)


@overload
def tile(A: XtructurableT, reps: int | Sequence[int]) -> XtructurableT:
    ...


@overload
def tile(A: ArrayLike, reps: int | Sequence[int]) -> jax.Array:
    ...


def tile(A, reps):
    if isinstance(A, Xtructurable):
        return _dc.tile(A, reps)
    return jnp.tile(A, reps)


@overload
def transpose(a: XtructurableT, axes: Sequence[int] | None = None) -> XtructurableT:
    ...


@overload
def transpose(a: ArrayLike, axes: Sequence[int] | None = None) -> jax.Array:
    ...


def transpose(a, axes: Sequence[int] | None = None):
    if isinstance(a, Xtructurable):
        return a.transpose(axes)
    return jnp.transpose(a, axes=axes)


@overload
def swapaxes(a: XtructurableT, axis1: int, axis2: int) -> XtructurableT:
    ...


@overload
def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> jax.Array:
    ...


def swapaxes(a, axis1: int, axis2: int):
    if isinstance(a, Xtructurable):
        return a.swapaxes(axis1, axis2)
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


@overload
def update_on_condition(
    dataclass_instance: XtructurableT,
    indices: Any,
    condition: ArrayLike,
    values_to_set: Any,
    *,
    unique_indices: bool = False,
) -> XtructurableT:
    ...


@overload
def update_on_condition(
    dataclass_instance: ArrayLike,
    indices: Any,
    condition: ArrayLike,
    values_to_set: Any,
    *,
    unique_indices: bool = False,
) -> jax.Array:
    ...


def update_on_condition(
    dataclass_instance, indices, condition, values_to_set, *, unique_indices: bool = False
):
    if isinstance(dataclass_instance, Xtructurable):
        if isinstance(values_to_set, Xtructurable):
            if jax.tree_util.tree_structure(values_to_set) != jax.tree_util.tree_structure(
                dataclass_instance
            ):
                raise TypeError(
                    "values_to_set must have the same tree structure as dataclass_instance."
                )
        return _dc.update_on_condition(
            dataclass_instance,
            indices,
            condition,
            values_to_set,
            unique_indices=unique_indices,
        )
    if isinstance(values_to_set, Xtructurable):
        raise TypeError("values_to_set must not be an xtructure dataclass when updating an array.")
    return _update_array_on_condition(
        dataclass_instance, indices, condition, values_to_set, unique_indices=unique_indices
    )


@overload
def expand_dims(a: XtructurableT, axis: int | Sequence[int]) -> XtructurableT:
    ...


@overload
def expand_dims(a: ArrayLike, axis: int | Sequence[int]) -> jax.Array:
    ...


def expand_dims(a, axis: int | Sequence[int]):
    if isinstance(a, Xtructurable):
        return _dc.expand_dims(a, axis=axis)
    return jnp.expand_dims(a, axis=axis)


@overload
def squeeze(a: XtructurableT, axis: int | Sequence[int] | None = None) -> XtructurableT:
    ...


@overload
def squeeze(a: ArrayLike, axis: int | Sequence[int] | None = None) -> jax.Array:
    ...


def squeeze(a, axis: int | Sequence[int] | None = None):
    if isinstance(a, Xtructurable):
        return _dc.squeeze(a, axis=axis)
    return jnp.squeeze(a, axis=axis)


@overload
def repeat(
    a: XtructurableT,
    repeats: Any,
    axis: int | None = None,
    *,
    total_repeat_length: None = None,
    out_sharding: None = None,
) -> XtructurableT:
    ...


@overload
def repeat(
    a: ArrayLike,
    repeats: Any,
    axis: int | None = None,
    *,
    total_repeat_length: int | None = None,
    out_sharding: Any = None,
) -> jax.Array:
    ...


def repeat(
    a,
    repeats,
    axis: int | None = None,
    *,
    total_repeat_length: int | None = None,
    out_sharding=None,
):
    if isinstance(a, Xtructurable):
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


@overload
def split(ary: XtructurableT, indices_or_sections: Any, axis: int = 0) -> list[XtructurableT]:
    ...


@overload
def split(ary: ArrayLike, indices_or_sections: Any, axis: int = 0) -> list[jax.Array]:
    ...


def split(ary, indices_or_sections, axis: int = 0):
    if isinstance(ary, Xtructurable):
        return _dc.split(ary, indices_or_sections, axis=axis)
    return list(jnp.split(ary, indices_or_sections, axis=axis))


@overload
def full_like(
    a: XtructurableT,
    fill_value: Any,
    dtype: None = None,
    shape: None = None,
    *,
    device: None = None,
    out_sharding: None = None,
) -> XtructurableT:
    ...


@overload
def full_like(
    a: ArrayLike,
    fill_value: Any,
    dtype: Any | None = None,
    shape: Any = None,
    *,
    device: Any = None,
    out_sharding: Any = None,
) -> jax.Array:
    ...


def full_like(
    a,
    fill_value,
    dtype: Any | None = None,
    shape: Any = None,
    *,
    device=None,
    out_sharding=None,
):
    if isinstance(a, Xtructurable):
        _reject_dataclass_kwargs(
            "full_like", dtype=dtype, shape=shape, device=device, out_sharding=out_sharding
        )
        return _dc.full_like(a, fill_value)
    return jnp.full_like(
        a,
        fill_value,
        **_like_kwargs(
            "full_like",
            jnp.full_like,
            dtype=dtype,
            shape=shape,
            device=device,
            out_sharding=out_sharding,
        ),
    )


@overload
def zeros_like(
    a: XtructurableT,
    dtype: None = None,
    shape: None = None,
    *,
    device: None = None,
    out_sharding: None = None,
) -> XtructurableT:
    ...


@overload
def zeros_like(
    a: ArrayLike,
    dtype: Any | None = None,
    shape: Any = None,
    *,
    device: Any = None,
    out_sharding: Any = None,
) -> jax.Array:
    ...


def zeros_like(
    a,
    dtype: Any | None = None,
    shape: Any = None,
    *,
    device=None,
    out_sharding=None,
):
    if isinstance(a, Xtructurable):
        _reject_dataclass_kwargs(
            "zeros_like",
            dtype=dtype,
            shape=shape,
            device=device,
            out_sharding=out_sharding,
        )
        return _dc.zeros_like(a)
    return jnp.zeros_like(
        a,
        **_like_kwargs(
            "zeros_like",
            jnp.zeros_like,
            dtype=dtype,
            shape=shape,
            device=device,
            out_sharding=out_sharding,
        ),
    )


@overload
def ones_like(
    a: XtructurableT,
    dtype: None = None,
    shape: None = None,
    *,
    device: None = None,
    out_sharding: None = None,
) -> XtructurableT:
    ...


@overload
def ones_like(
    a: ArrayLike,
    dtype: Any | None = None,
    shape: Any = None,
    *,
    device: Any = None,
    out_sharding: Any = None,
) -> jax.Array:
    ...


def ones_like(
    a,
    dtype: Any | None = None,
    shape: Any = None,
    *,
    device=None,
    out_sharding=None,
):
    if isinstance(a, Xtructurable):
        _reject_dataclass_kwargs(
            "ones_like",
            dtype=dtype,
            shape=shape,
            device=device,
            out_sharding=out_sharding,
        )
        return _dc.ones_like(a)
    return jnp.ones_like(
        a,
        **_like_kwargs(
            "ones_like",
            jnp.ones_like,
            dtype=dtype,
            shape=shape,
            device=device,
            out_sharding=out_sharding,
        ),
    )


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
