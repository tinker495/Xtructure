"""Logical helpers for xtructure dataclasses."""

from __future__ import annotations

from typing import Any, Union

import jax
import jax.numpy as jnp

from ...xtructure_decorators import Xtructurable
from ...xtructure_numpy.array_ops import _update_array_on_condition, _where_no_broadcast


def where(condition: jnp.ndarray, x: Xtructurable, y: Union[Xtructurable, Any]) -> Xtructurable:
    """Apply jnp.where across every field of a dataclass."""
    condition_array = jnp.asarray(condition, dtype=jnp.bool_)

    def _align_condition(target_shape: tuple[int, ...]) -> jnp.ndarray:
        if condition_array.shape == target_shape:
            return condition_array
        try:
            return jnp.broadcast_to(condition_array, target_shape)
        except ValueError as err:
            raise ValueError(
                f"`condition` with shape {condition_array.shape} cannot be broadcast to target shape {target_shape}."
            ) from err

    y_leaves = jax.tree_util.tree_leaves(y)
    if len(y_leaves) > 1 or (len(y_leaves) == 1 and hasattr(y, "__dataclass_fields__")):

        def _apply_dataclass_where(x_field, y_field):
            cond = _align_condition(x_field.shape)
            y_array = jnp.asarray(y_field)
            if y_array.shape != x_field.shape:
                try:
                    y_array = jnp.broadcast_to(y_array, x_field.shape)
                except ValueError as err:
                    raise ValueError(
                        f"`y` field with shape {y_array.shape} cannot be"
                        "broadcast to match `x` field shape {x_field.shape}."
                        f"Original `y` shape: {y_array.shape}, `x` shape: {x_field.shape}."
                    ) from err
            target_dtype = jnp.result_type(x_field.dtype, y_array.dtype)
            return _where_no_broadcast(
                cond,
                jnp.asarray(x_field, dtype=target_dtype),
                jnp.asarray(y_array, dtype=target_dtype),
            )

        return jax.tree_util.tree_map(_apply_dataclass_where, x, y)

    scalar_value = jnp.asarray(y)

    def _apply_scalar_where(x_field):
        cond = _align_condition(x_field.shape)
        try:
            y_array = jnp.broadcast_to(scalar_value, x_field.shape)
        except ValueError as err:
            raise ValueError(
                f"`y` value with shape {scalar_value.shape} cannot be"
                "broadcast to match `x` field shape {x_field.shape}."
                f"Original `y` shape: {scalar_value.shape}, `x` shape: {x_field.shape}."
            ) from err
        target_dtype = jnp.result_type(x_field.dtype, y_array.dtype)
        return _where_no_broadcast(
            cond,
            jnp.asarray(x_field, dtype=target_dtype),
            jnp.asarray(y_array, dtype=target_dtype),
        )

    return jax.tree_util.tree_map(_apply_scalar_where, x)


def where_no_broadcast(
    condition: Union[jnp.ndarray, Xtructurable],
    x: Xtructurable,
    y: Xtructurable,
) -> Xtructurable:
    """Apply a strict where across dataclass fields without implicit broadcasting."""
    if type(x) is not type(y):
        raise TypeError(
            "`x` and `y` must be instances of the same dataclass for where_no_broadcast."
        )

    condition_is_dataclass = hasattr(condition, "__dataclass_fields__")

    if condition_is_dataclass:
        condition_structure = jax.tree_util.tree_structure(condition)
        x_structure = jax.tree_util.tree_structure(x)
        if condition_structure != x_structure:
            raise TypeError(
                "`condition` must share the same dataclass structure as `x` and `y` "
                "when provided as a dataclass."
            )

        return jax.tree_util.tree_map(
            lambda cond_field, x_field, y_field: _where_no_broadcast(cond_field, x_field, y_field),
            condition,
            x,
            y,
        )

    condition_array = jnp.asarray(condition, dtype=jnp.bool_)
    return jax.tree_util.tree_map(
        lambda x_field, y_field: _where_no_broadcast(condition_array, x_field, y_field),
        x,
        y,
    )


def update_on_condition(
    dataclass_instance: Xtructurable,
    indices: Union[jnp.ndarray, tuple[jnp.ndarray, ...]],
    condition: jnp.ndarray,
    values_to_set: Union[Xtructurable, Any],
) -> Xtructurable:
    """Condtionally update fields with values, ensuring first True wins for duplicates."""
    values_leaves = jax.tree_util.tree_leaves(values_to_set)
    if len(values_leaves) > 1 or (
        len(values_leaves) == 1 and hasattr(values_to_set, "__dataclass_fields__")
    ):
        return jax.tree_util.tree_map(
            lambda field, values_field: _update_array_on_condition(
                field, indices, condition, values_field
            ),
            dataclass_instance,
            values_to_set,
        )
    return jax.tree_util.tree_map(
        lambda field: _update_array_on_condition(field, indices, condition, values_to_set),
        dataclass_instance,
    )
