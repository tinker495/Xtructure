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


def _compute_unique_mask_from_uint32eds(
    uint32eds: chex.Array,
    filled: chex.Array,
    unique_key: chex.Array | None,
) -> tuple[chex.Array, chex.Array]:
    filled = jnp.asarray(filled, dtype=jnp.bool_)
    batch_len = filled.shape[0]

    if uint32eds.ndim == 1:
        uint32eds = uint32eds[:, None]

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
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Batched twin of :func:`get_new_idx_byterized`.

    Consumes the **Instance Layout-aware** ``.hash_with_uint32ed`` BATCHED
    surface (returns ``((n,), (n, lanes))``) and the row-wise hash reducer to
    avoid the per-row ``jax.vmap(get_new_idx_byterized, in_axes=(0, None, None))``
    wrapper that BATCHED parallel insert / lookup paths previously needed.
    """
    hash_value, uint32ed = inputs.hash_with_uint32ed(seed)  # ((n,), (n, lanes))
    seed_u32 = jnp.asarray(seed, dtype=jnp.uint32)
    secondary_seed = jnp.bitwise_xor(seed_u32, DOUBLE_HASH_SECONDARY_DELTA)
    secondary_hash = hash_fast_uint32ed_batched(uint32ed, secondary_seed)  # (n,)
    idx = _modulus_reduce(hash_value, modulus)  # (n,)
    step = _normalize_probe_step(secondary_hash, modulus)  # (n,)
    length = jnp.uint32(uint32ed.shape[1])  # scalar (per-row lane count)
    fingerprint = _mix_fingerprint(hash_value, secondary_hash, length)  # (n,)
    return idx, step, uint32ed, fingerprint
