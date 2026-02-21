"""Legacy unique_mask implementation for comparison."""

from __future__ import annotations

from typing import Any, Callable, Union

import jax
import jax.numpy as jnp

from ....xtructure_decorators import Xtructurable
from ....xtructure_numpy.array_ops import _where_no_broadcast


def unique_mask_legacy(
    val: Xtructurable,
    key: jnp.ndarray | None = None,
    filled: jnp.ndarray | None = None,
    key_fn: Callable[[Any], jnp.ndarray] | None = None,
    batch_len: int | None = None,
    return_index: bool = False,
    return_inverse: bool = False,
) -> Union[jnp.ndarray, tuple]:
    """Legacy implementation using jnp.unique + scatter reduction."""
    if key_fn is None:

        def key_fn(x):
            return x.uint32ed

    try:
        hash_bytes = jax.vmap(key_fn)(val)
    except Exception as e:
        raise ValueError(f"key_fn failed to generate hashable keys: {e}")

    if batch_len is None:
        batch_len = val.shape.batch[0]

    if batch_len == 0:
        final_mask = jnp.zeros((0,), dtype=jnp.bool_)
        if not return_index and not return_inverse:
            return final_mask
        returns = (final_mask,)
        if return_index:
            returns += (jnp.zeros((0,), dtype=jnp.int32),)
        if return_inverse:
            returns += (jnp.zeros((0,), dtype=jnp.int32),)
        return returns

    unique_keys = hash_bytes
    if filled is not None:
        flat_keys = unique_keys.reshape(batch_len, -1)
        filled_col = filled.reshape(batch_len, 1).astype(flat_keys.dtype)
        unique_keys = jnp.concatenate([flat_keys, filled_col], axis=1)

    _, unique_indices, inv = jnp.unique(
        unique_keys,
        axis=0,
        size=batch_len,
        return_index=True,
        return_inverse=True,
    )

    batch_idx = jnp.arange(batch_len, dtype=jnp.int32)

    if key is None:
        final_mask = jnp.zeros(batch_len, dtype=jnp.bool_).at[unique_indices].set(True)
        if filled is not None:
            final_mask = jnp.logical_and(final_mask, filled)
    else:
        if filled is not None:
            inf_fill = jnp.full_like(key, jnp.inf)
            masked_key = _where_no_broadcast(filled, key, inf_fill)
        else:
            masked_key = key

        min_costs_per_group = jnp.full((batch_len,), jnp.inf, dtype=key.dtype)
        min_costs_per_group = min_costs_per_group.at[inv].min(masked_key)

        min_cost_for_each_item = min_costs_per_group[inv]
        is_min_cost = masked_key == min_cost_for_each_item

        if filled is not None:
            can_be_considered = jnp.logical_and(is_min_cost, filled)
            fallback_idx = jnp.full_like(batch_idx, batch_len)
            indices_to_consider = _where_no_broadcast(
                can_be_considered, batch_idx, fallback_idx
            )
        else:
            fallback_idx = jnp.full_like(batch_idx, batch_len)
            indices_to_consider = _where_no_broadcast(
                is_min_cost, batch_idx, fallback_idx
            )

        winning_indices_per_group = jnp.full((batch_len,), batch_len, dtype=jnp.int32)
        winning_indices_per_group = winning_indices_per_group.at[inv].min(
            indices_to_consider
        )

        winning_index_for_each_item = winning_indices_per_group[inv]
        final_mask = batch_idx == winning_index_for_each_item

        if filled is None:
            is_valid = key < jnp.inf
            final_mask = jnp.logical_and(final_mask, is_valid)

        if return_index:
            unique_group_ids, _ = jnp.unique(inv, size=batch_len, return_index=True)
            unique_indices = winning_indices_per_group[unique_group_ids]

    if not return_index and not return_inverse:
        return final_mask

    returns = (final_mask,)
    if return_index:
        returns += (unique_indices,)
    if return_inverse:
        returns += (inv,)

    return returns
