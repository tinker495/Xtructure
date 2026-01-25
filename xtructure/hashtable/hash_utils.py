"""Hash helpers for bucketed double hashing."""
from __future__ import annotations

from typing import cast

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable
from ..core.xtructure_decorators.hash import uint32ed_to_hash
from .constants import DOUBLE_HASH_SECONDARY_DELTA, SIZE_DTYPE


def _first_occurrence_mask(
    values: chex.Array, active: chex.Array, sentinel: chex.Array
) -> chex.Array:
    active = jnp.asarray(active, dtype=jnp.bool_)
    values = jnp.asarray(values, dtype=jnp.uint32)
    sentinel = jnp.asarray(sentinel, dtype=jnp.uint32)

    safe_values = jnp.where(active, values, sentinel)
    _, unique_indices = jnp.unique(
        safe_values,
        size=values.shape[0],
        return_index=True,
        return_inverse=False,
        fill_value=sentinel,
    )
    mask = jnp.zeros_like(active, dtype=jnp.bool_).at[unique_indices].set(True)
    return jnp.logical_and(mask, active)


def _compute_unique_mask_from_uint32eds(
    uint32eds: chex.Array,
    filled: chex.Array,
    unique_key: chex.Array | None,
) -> tuple[chex.Array, chex.Array]:
    if uint32eds.ndim == 0:
        raise ValueError("uint32eds must be at least rank-1.")

    batch_len = int(uint32eds.shape[0])
    filled = cast(jax.Array, jnp.asarray(filled, dtype=jnp.bool_))
    if filled.ndim == 0:
        filled = cast(jax.Array, jnp.full((batch_len,), filled, dtype=jnp.bool_))
    elif filled.shape[0] != batch_len:
        raise ValueError("filled must match uint32eds leading dimension.")

    if uint32eds.ndim == 1:
        uint32eds = uint32eds[:, None]

    # NOTE: `jnp.unique(axis=0)` can be very expensive on GPU for wide rows.
    # We instead perform a lexicographic sort over the row words and compute
    # group representatives via scatter-min reductions.

    sentinel = jnp.uint32(0xFFFFFFFF)
    sentinel_row = jnp.full_like(uint32eds, sentinel)
    safe_uint32eds = jnp.where(filled[:, None], uint32eds, sentinel_row)

    indices = jnp.arange(batch_len, dtype=jnp.int32)

    # Sort by the row words (lexicographic), stable for deterministic ties.
    if safe_uint32eds.ndim != 2:
        raise ValueError("uint32eds must be rank-2 after normalization.")

    word_count = int(safe_uint32eds.shape[1])
    sort_keys = tuple(safe_uint32eds[:, i] for i in range(word_count))
    sorted_out = cast(
        tuple[jax.Array, ...],
        jax.lax.sort(
            (*sort_keys, indices),
            dimension=0,
            is_stable=True,
            num_keys=word_count,
        ),
    )
    sorted_words = sorted_out[:-1]
    sorted_indices = sorted_out[-1]

    sorted_filled = filled[sorted_indices]

    # Group boundaries: row changes between adjacent sorted elements.
    if batch_len == 0:
        representative_indices = jnp.zeros((0,), dtype=jnp.int32)
        unique_mask = jnp.zeros((0,), dtype=jnp.bool_)
        return unique_mask, representative_indices

    row_changed = jnp.zeros((batch_len - 1,), dtype=jnp.bool_)
    for w in sorted_words:
        w_arr = cast(jax.Array, w)
        row_changed = jnp.logical_or(row_changed, w_arr[1:] != w_arr[:-1])

    is_group_start = jnp.concatenate([jnp.array([True]), row_changed], axis=0)
    group_id = jnp.cumsum(is_group_start.astype(jnp.int32)) - jnp.int32(1)

    batch_len_i32 = jnp.int32(batch_len)

    if unique_key is not None:
        unique_key_arr = cast(jax.Array, jnp.asarray(unique_key))
        if unique_key_arr.ndim == 0:
            unique_key_arr = cast(jax.Array, jnp.full((batch_len,), unique_key_arr))
        elif unique_key_arr.shape[0] != batch_len:
            raise ValueError("unique_key must match uint32eds leading dimension.")

        sorted_unique_key = unique_key_arr[sorted_indices]
        masked_key = jnp.where(sorted_filled, sorted_unique_key, jnp.inf)
        min_keys = (
            jnp.full((batch_len,), jnp.inf, dtype=masked_key.dtype).at[group_id].min(masked_key)
        )
        candidate_indices = jnp.where(
            masked_key == min_keys[group_id],
            sorted_indices,
            batch_len_i32,
        )
    else:
        candidate_indices = jnp.where(sorted_filled, sorted_indices, batch_len_i32)

    representative_per_group = (
        jnp.full((batch_len,), batch_len_i32, dtype=jnp.int32).at[group_id].min(candidate_indices)
    )
    representative_per_group = jnp.where(
        representative_per_group == batch_len_i32, jnp.int32(0), representative_per_group
    )
    representative_sorted = representative_per_group[group_id]

    representative_indices = cast(jax.Array, jnp.zeros((batch_len,), dtype=jnp.int32))
    representative_indices = cast(
        jax.Array,
        representative_indices.at[sorted_indices].set(representative_sorted),
    )
    representative_indices = cast(
        jax.Array,
        jnp.where(filled, representative_indices, jnp.int32(0)),
    )

    unique_mask = jnp.logical_and(filled, indices == representative_indices)
    return unique_mask, cast(jax.Array, representative_indices)


def _normalize_probe_step(step: chex.Array, modulus: int) -> chex.Array:
    step_u32 = jnp.asarray(step, dtype=SIZE_DTYPE)
    modulus_u32 = jnp.asarray(modulus, dtype=SIZE_DTYPE)
    modulus_u32 = jnp.maximum(modulus_u32, SIZE_DTYPE(1))
    mask = modulus_u32 - SIZE_DTYPE(1)
    is_pow2 = jnp.logical_and(modulus_u32 > 0, (modulus_u32 & mask) == 0)
    step_u32 = cast(jax.Array, jnp.where(is_pow2, step_u32 & mask, step_u32 % modulus_u32))
    step_u32 = cast(jax.Array, jnp.bitwise_or(step_u32, SIZE_DTYPE(1)))
    return cast(jax.Array, step_u32)


def get_new_idx_from_uint32ed(
    input_uint32ed: chex.Array,
    modulus: int,
    seed: int,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Calculate a new hash bucket index, probe step, and 2x32-bit tags from a uint32ed."""
    seed_u32 = jnp.asarray(seed, dtype=jnp.uint32)
    primary_hash = uint32ed_to_hash(input_uint32ed, seed_u32)
    secondary_seed = jnp.bitwise_xor(seed_u32, DOUBLE_HASH_SECONDARY_DELTA)
    secondary_hash = uint32ed_to_hash(input_uint32ed, secondary_seed)
    modulus_u32 = jnp.asarray(modulus, dtype=SIZE_DTYPE)
    modulus_u32 = jnp.maximum(modulus_u32, SIZE_DTYPE(1))
    mask = modulus_u32 - SIZE_DTYPE(1)
    is_pow2 = jnp.logical_and(modulus_u32 > 0, (modulus_u32 & mask) == 0)
    index = jax.lax.select(
        is_pow2, jnp.asarray(primary_hash, dtype=SIZE_DTYPE) & mask, primary_hash % modulus_u32
    )
    step = _normalize_probe_step(secondary_hash, modulus)
    return index, step, primary_hash, secondary_hash


def get_new_idx_byterized(
    input: Xtructurable,
    modulus: int,
    seed: int,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Hash a Xtructurable and return index, step, uint32ed, and 2x32-bit tags."""
    hash_value, uint32ed = input.hash_with_uint32ed(seed)
    hash_value = jnp.asarray(hash_value, dtype=jnp.uint32)
    seed_u32 = jnp.asarray(seed, dtype=jnp.uint32)
    secondary_seed = jnp.bitwise_xor(seed_u32, DOUBLE_HASH_SECONDARY_DELTA)
    secondary_hash = uint32ed_to_hash(uint32ed, secondary_seed)
    modulus_u32 = jnp.asarray(modulus, dtype=SIZE_DTYPE)
    modulus_u32 = jnp.maximum(modulus_u32, SIZE_DTYPE(1))
    mask = modulus_u32 - SIZE_DTYPE(1)
    is_pow2 = jnp.logical_and(modulus_u32 > 0, (modulus_u32 & mask) == 0)
    idx = jax.lax.select(
        is_pow2, jnp.asarray(hash_value, dtype=SIZE_DTYPE) & mask, hash_value % modulus_u32
    )
    step = _normalize_probe_step(secondary_hash, modulus)
    return idx, step, uint32ed, hash_value, secondary_hash
