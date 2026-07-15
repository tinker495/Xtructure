"""Hash helpers for bucketed double hashing."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from ..core.dtype_facts import SIZE_DTYPE
from ..core.protocol import Xtructurable
from ..core.xtructure_decorators.pytree_adapters.hash import (
    hash_fast_uint32ed,
    hash_fast_uint32ed_batched,
)
from .constants import (
    DOUBLE_HASH_SECONDARY_DELTA,
    FINGERPRINT_MIX_CONSTANT_A,
    FINGERPRINT_MIX_CONSTANT_B,
)
from .types import HashTableProbe

# Seed for the intra-batch dedup row hash. Independent of the table seed —
# dedup grouping is per-batch and never persisted.
_UNIQUE_HASH_SEED = jnp.uint32(0xA5A5_A5A5)


def _mix_fingerprint(primary: chex.Array, secondary: chex.Array, length: chex.Array) -> chex.Array:
    mix = jnp.asarray(primary, dtype=jnp.uint32)
    secondary = jnp.asarray(secondary, dtype=jnp.uint32)
    length = jnp.asarray(length, dtype=jnp.uint32)

    mix ^= jnp.uint32(0x9E3779B9)
    mix = jnp.uint32(
        mix + secondary * FINGERPRINT_MIX_CONSTANT_A + length * FINGERPRINT_MIX_CONSTANT_B
    )
    mix ^= mix >> jnp.uint32(16)
    mix *= jnp.uint32(0x7FEB352D)
    mix ^= mix >> jnp.uint32(15)
    return mix


def _unique_mask_from_group_ids(
    inverse: chex.Array,
    filled: chex.Array,
    unique_key: chex.Array | None,
) -> tuple[chex.Array, chex.Array]:
    """Representative selection shared by the exact and fast dedup paths.

    ``inverse`` assigns one group id per batch row; rows in the same group are
    duplicates. The representative is the lowest batch index in the group
    (among minimum-``unique_key`` holders when a key is given).
    """
    batch_len = filled.shape[0]
    indices = jnp.arange(batch_len, dtype=jnp.int32)

    if unique_key is not None:
        masked_key = jnp.where(filled, unique_key, jnp.inf)
        min_keys = (
            jnp.full((batch_len,), jnp.inf, dtype=masked_key.dtype).at[inverse].min(masked_key)
        )
        candidate_indices = jnp.where(masked_key == min_keys[inverse], indices, batch_len)
    else:
        candidate_indices = jnp.where(filled, indices, batch_len)

    representative_per_group = (
        jnp.full((batch_len,), batch_len, dtype=jnp.int32).at[inverse].min(candidate_indices)
    )
    representative_per_group = jnp.where(
        representative_per_group == batch_len, 0, representative_per_group
    )

    representative_indices = representative_per_group[inverse]
    representative_indices = jnp.where(filled, representative_indices, 0)

    unique_mask = jnp.logical_and(filled, indices == representative_indices)
    return unique_mask, representative_indices


def _unique_groups_exact(uint32eds: chex.Array, filled: chex.Array) -> chex.Array:
    """Exact row grouping via full-row unique (multi-column lexsort)."""
    batch_len = filled.shape[0]
    sentinel_row = jnp.full_like(uint32eds, jnp.uint32(0xFFFFFFFF))
    safe_uint32eds = jnp.where(filled[:, None], uint32eds, sentinel_row)
    fill_row = jnp.full((uint32eds.shape[1],), jnp.uint32(0xFFFFFFFF))
    _, inverse = jnp.unique(
        safe_uint32eds,
        axis=0,
        size=batch_len,
        fill_value=fill_row,
        return_inverse=True,
    )
    return inverse


def _compute_unique_mask_from_uint32eds(
    uint32eds: chex.Array,
    filled: chex.Array,
    unique_key: chex.Array | None,
    row_hash: chex.Array | None = None,
) -> tuple[chex.Array, chex.Array]:
    filled = jnp.asarray(filled, dtype=jnp.bool_)
    batch_len = filled.shape[0]

    if uint32eds.ndim == 1:
        uint32eds = uint32eds[:, None]

    # Fast path: group by one 32-bit row hash (u32 keys sort ~L× cheaper than
    # the full-row lexsort), then verify each row equals its group's first
    # member. Row-equal rows always hash equal, so groups can only be too
    # coarse — and any coarseness is caught by the verification and routed to
    # the exact full-row unique. Dedup stays EXACT for adversarial inputs.
    if row_hash is None:
        row_hash = hash_fast_uint32ed_batched(uint32eds, _UNIQUE_HASH_SEED)
    row_hash = jnp.asarray(row_hash, dtype=jnp.uint32)
    unfilled_flag = jnp.where(filled, jnp.uint32(0), jnp.uint32(1))
    indices = jnp.arange(batch_len, dtype=jnp.int32)

    sorted_flag, sorted_hash, sorted_idx = jax.lax.sort(
        (unfilled_flag, row_hash, indices.astype(jnp.uint32)), dimension=0, num_keys=3
    )
    new_group = jnp.concatenate(
        [
            jnp.array([True]),
            jnp.logical_or(
                sorted_flag[1:] != sorted_flag[:-1], sorted_hash[1:] != sorted_hash[:-1]
            ),
        ],
        axis=0,
    )
    gid_sorted = jnp.cumsum(new_group.astype(jnp.int32)) - 1
    inverse_fast = jnp.zeros((batch_len,), dtype=jnp.int32).at[sorted_idx].set(gid_sorted)

    has_multi_filled_group = jnp.any(
        jnp.logical_and(jnp.logical_not(new_group[1:]), sorted_flag[1:] == jnp.uint32(0))
    )

    def _verified_inverse() -> chex.Array:
        first_member = (
            jnp.full((batch_len,), batch_len, dtype=jnp.int32).at[inverse_fast].min(indices)
        )
        rep_rows = uint32eds[first_member[inverse_fast]]
        row_matches_rep = jnp.all(uint32eds == rep_rows, axis=1)
        hash_collision = jnp.any(jnp.logical_and(filled, jnp.logical_not(row_matches_rep)))
        return jax.lax.cond(
            hash_collision,
            lambda: _unique_groups_exact(uint32eds, filled),
            lambda: inverse_fast,
        )

    inverse = jax.lax.cond(has_multi_filled_group, _verified_inverse, lambda: inverse_fast)
    return _unique_mask_from_group_ids(inverse, filled, unique_key)


def _modulus_reduce(value: chex.Array, modulus: int) -> chex.Array:
    """Reduce ``value`` modulo ``modulus`` using a bitmask fast path for powers of two."""
    modulus_u32 = jnp.asarray(modulus, dtype=SIZE_DTYPE)
    modulus_u32 = jnp.maximum(modulus_u32, SIZE_DTYPE(1))
    mask = modulus_u32 - SIZE_DTYPE(1)
    is_pow2 = jnp.logical_and(modulus_u32 > 0, (modulus_u32 & mask) == 0)
    value_u32 = jnp.asarray(value, dtype=SIZE_DTYPE)
    return jax.lax.select(is_pow2, value_u32 & mask, value_u32 % modulus_u32)


def _normalize_probe_step(step: chex.Array, modulus: int) -> chex.Array:
    step = _modulus_reduce(step, modulus)
    step = jnp.bitwise_or(step, SIZE_DTYPE(1))
    return step


def get_new_idx_byterized(
    input: Xtructurable,
    modulus: int,
    seed: int,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Hash a Xtructurable and return index, step, byte representation, and fingerprint."""
    hash_value, uint32ed = input.hash_with_uint32ed(seed)
    seed_u32 = jnp.asarray(seed, dtype=jnp.uint32)
    secondary_seed = jnp.bitwise_xor(seed_u32, DOUBLE_HASH_SECONDARY_DELTA)
    secondary_hash = hash_fast_uint32ed(uint32ed, secondary_seed)
    idx = _modulus_reduce(hash_value, modulus)
    step = _normalize_probe_step(secondary_hash, modulus)
    length = jnp.uint32(uint32ed.size)
    fingerprint = _mix_fingerprint(hash_value, secondary_hash, length)
    return idx, step, uint32ed, fingerprint


def get_new_idx_byterized_batched(
    inputs: Xtructurable,
    modulus: int,
    seed: int,
) -> HashTableProbe:
    """Batched twin of :func:`get_new_idx_byterized`.

    Consumes the **Instance Layout-aware** ``.hash_with_uint32ed`` BATCHED
    surface (returns ``((n,), (n, lanes))``) and the row-wise hash reducer to
    avoid the per-row ``jax.vmap(get_new_idx_byterized, in_axes=(0, None, None))``
    wrapper that BATCHED parallel insert / lookup paths previously needed.

    Returns a :class:`HashTableProbe` bundling ``(index, step, uint32ed,
    fingerprint)`` so parallel lookup and parallel insert can consume one shared
    hash pass. This is the single canonical producer of that probe.
    """
    hash_value, uint32ed = inputs.hash_with_uint32ed(seed)  # ((n,), (n, lanes))
    seed_u32 = jnp.asarray(seed, dtype=jnp.uint32)
    secondary_seed = jnp.bitwise_xor(seed_u32, DOUBLE_HASH_SECONDARY_DELTA)
    secondary_hash = hash_fast_uint32ed_batched(uint32ed, secondary_seed)  # (n,)
    idx = _modulus_reduce(hash_value, modulus)  # (n,)
    step = _normalize_probe_step(secondary_hash, modulus)  # (n,)
    length = jnp.uint32(uint32ed.shape[1])  # scalar (per-row lane count)
    fingerprint = _mix_fingerprint(hash_value, secondary_hash, length)  # (n,)
    return HashTableProbe(index=idx, step=step, uint32ed=uint32ed, fingerprint=fingerprint)
