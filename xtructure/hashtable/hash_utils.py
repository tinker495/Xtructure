"""Hash helpers for bucketed double hashing."""
from __future__ import annotations

import os
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from jax import lax

from ..core import Xtructurable
from ..core.xtructure_decorators.hash import _mix_fingerprint, uint32ed_to_hash
from .constants import DOUBLE_HASH_SECONDARY_DELTA, SIZE_DTYPE


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value.")


def _parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"", "none", "auto"}:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


_DEDUPE_MODE_RAW = os.environ.get("XTRUCTURE_HASHTABLE_DEDUPE_MODE", "safe").strip().lower()

_SORT_BACKEND = os.environ.get("XTRUCTURE_HASHTABLE_SORT_BACKEND", "stable_argsort").strip().lower()
if _SORT_BACKEND not in {"stable_argsort", "lax_unstable", "lax_stable"}:
    raise ValueError(
        "Invalid XTRUCTURE_HASHTABLE_SORT_BACKEND. Expected one of: stable_argsort, lax_unstable, lax_stable."
    )

# Dedupe mode semantics:
# - "safe" (default): exact for small keys, signature sort for wide keys with collision detection
#   and fallback to full-row sort when needed.
# - "exact": always full-row sort.
# - "approx": signature-only for wide keys; may drop distinct inputs on signature collision.
#
# Backward-compatibility: "fast" is treated as "safe" (correctness-preserving).
if _DEDUPE_MODE_RAW == "fast":
    _DEDUPE_MODE = "safe"
else:
    _DEDUPE_MODE = _DEDUPE_MODE_RAW

if _DEDUPE_MODE not in {"approx", "safe", "exact"}:
    raise ValueError(
        "Invalid XTRUCTURE_HASHTABLE_DEDUPE_MODE. Expected one of: approx, safe, exact (or fast)."
    )


# Exported for call-site specialization (read at import time).
HASHTABLE_DEDUPE_MODE = _DEDUPE_MODE


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

    if batch_len == 0:
        representative_indices = jnp.zeros((0,), dtype=jnp.int32)
        unique_mask = jnp.zeros((0,), dtype=jnp.bool_)
        return unique_mask, representative_indices

    # NOTE: Sorting by full uint32ed rows can be very expensive on GPU for wide keys.
    # For small keys we sort by the full row (exact). For wide keys we reduce each
    # row into a fixed-width (128-bit) signature (4x uint32) and sort/group on that.

    keys = jnp.asarray(uint32eds, dtype=jnp.uint32)
    if keys.ndim != 2:
        raise ValueError("uint32eds must be rank-2 after normalization.")

    word_count = int(keys.shape[1])
    indices = jnp.arange(batch_len, dtype=jnp.int32)

    def _stable_sort_perm(perm_in: jax.Array, key_1d: jax.Array) -> jax.Array:
        order = jnp.argsort(key_1d[perm_in], stable=True)
        return perm_in[order]

    def _full_row_sort() -> tuple[jax.Array, jax.Array]:
        if word_count == 0:
            sorted_indices = indices
            row_changed = jnp.zeros((batch_len - 1,), dtype=jnp.bool_)
            return cast(jax.Array, sorted_indices), cast(jax.Array, row_changed)

        sentinel = jnp.broadcast_to(jnp.uint32(0xFFFFFFFF), (batch_len,))
        sort_keys = []
        for i in range(word_count):
            sort_keys.append(
                jnp.asarray(jax.lax.select(filled, keys[:, i], sentinel), dtype=jnp.uint32)
            )

        perm = indices
        # Primary key is column 0, then 1, ...; apply stable sorts from least-significant.
        for k in reversed(sort_keys):
            perm = _stable_sort_perm(perm, cast(jax.Array, k))

        sorted_indices = cast(jax.Array, perm)
        sorted_keys = [cast(jax.Array, k)[sorted_indices] for k in sort_keys]

        row_changed = jnp.zeros((batch_len - 1,), dtype=jnp.bool_)
        for i in range(word_count):
            k = cast(jax.Array, sorted_keys[i])
            row_changed = jnp.logical_or(row_changed, k[1:] != k[:-1])
        return cast(jax.Array, sorted_indices), cast(jax.Array, row_changed)

    if _DEDUPE_MODE == "exact" or word_count <= 8:
        sorted_indices, row_changed = _full_row_sort()
    else:
        # Wide-key path: sort/group on a 128-bit signature.
        if word_count == 0:
            h1 = jnp.zeros((batch_len,), dtype=jnp.uint32)
            h2 = jnp.zeros((batch_len,), dtype=jnp.uint32)
            h3 = jnp.zeros((batch_len,), dtype=jnp.uint32)
            h4 = jnp.zeros((batch_len,), dtype=jnp.uint32)
        else:
            # NOTE: do not use Python loops over word_count; it can be huge for wide keys
            # and causes XLA graph explosion at trace time.
            h1 = keys[:, 0]
            h2 = keys[:, 0]
            h3 = keys[:, 0]
            h4 = keys[:, 0]

            c1 = jnp.uint32(0x9E3779B1)
            c2 = jnp.uint32(0x85EBCA6B)
            c3 = jnp.uint32(0xC2B2AE35)
            c4 = jnp.uint32(0x278DDE6E)

            def _sig_body(i, carry):
                hh1, hh2, hh3, hh4 = carry
                col = lax.dynamic_index_in_dim(keys, i, axis=1, keepdims=False)
                hh1 = hh1 * c1 + col
                hh2 = hh2 * c2 + col
                hh3 = jnp.bitwise_xor(hh3, col) * c3
                hh4 = jnp.bitwise_xor(hh4, col) * c4
                return hh1, hh2, hh3, hh4

            h1, h2, h3, h4 = lax.fori_loop(1, word_count, _sig_body, (h1, h2, h3, h4))

        sentinel = jnp.broadcast_to(jnp.uint32(0xFFFFFFFF), h1.shape)
        h1 = jnp.asarray(jax.lax.select(filled, h1, sentinel), dtype=jnp.uint32)
        h2 = jnp.asarray(jax.lax.select(filled, h2, sentinel), dtype=jnp.uint32)
        h3 = jnp.asarray(jax.lax.select(filled, h3, sentinel), dtype=jnp.uint32)
        h4 = jnp.asarray(jax.lax.select(filled, h4, sentinel), dtype=jnp.uint32)

        perm_sig = indices
        perm_sig = _stable_sort_perm(perm_sig, h4)
        perm_sig = _stable_sort_perm(perm_sig, h3)
        perm_sig = _stable_sort_perm(perm_sig, h2)
        perm_sig = _stable_sort_perm(perm_sig, h1)

        sorted_indices_sig = cast(jax.Array, perm_sig)
        sorted_h1 = h1[sorted_indices_sig]
        sorted_h2 = h2[sorted_indices_sig]
        sorted_h3 = h3[sorted_indices_sig]
        sorted_h4 = h4[sorted_indices_sig]

        row_changed_sig = jnp.logical_or(
            sorted_h1[1:] != sorted_h1[:-1], sorted_h2[1:] != sorted_h2[:-1]
        )
        row_changed_sig = jnp.logical_or(row_changed_sig, sorted_h3[1:] != sorted_h3[:-1])
        row_changed_sig = jnp.logical_or(row_changed_sig, sorted_h4[1:] != sorted_h4[:-1])

        if _DEDUPE_MODE == "safe":
            sorted_filled_sig = filled[sorted_indices_sig]
            same_sig = jnp.logical_not(row_changed_sig)
            same_sig = jnp.logical_and(same_sig, sorted_filled_sig[1:])
            same_sig = jnp.logical_and(same_sig, sorted_filled_sig[:-1])
            has_sig_dups = jnp.any(same_sig)

            def _check_collision(_):
                lhs = keys[sorted_indices_sig[1:]]
                rhs = keys[sorted_indices_sig[:-1]]
                adj_equal = jnp.all(lhs == rhs, axis=1)
                collision = jnp.any(jnp.logical_and(same_sig, jnp.logical_not(adj_equal)))
                return collision

            collision = jax.lax.cond(
                has_sig_dups,
                _check_collision,
                lambda _: jnp.bool_(False),
                operand=None,
            )

            sorted_indices, row_changed = jax.lax.cond(
                collision,
                lambda _: _full_row_sort(),
                lambda _: (sorted_indices_sig, row_changed_sig),
                operand=None,
            )
        else:
            sorted_indices, row_changed = sorted_indices_sig, row_changed_sig

    sorted_filled = filled[sorted_indices]

    # Group boundaries: key/signature changes between adjacent sorted elements.
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


def _compute_unique_mask_from_hash_pairs(
    primary_hashes: chex.Array,
    secondary_hashes: chex.Array,
    filled: chex.Array,
    unique_key: chex.Array | None,
    *,
    uint32eds: chex.Array | None = None,
) -> tuple[chex.Array, chex.Array]:
    """Compute an in-batch uniqueness mask using a (primary, secondary) hash pair.

    In safe mode (default), this detects hash-pair collisions using `uint32eds`
    and falls back to an exact full-row dedupe when needed.

    Args:
        primary_hashes: (N,) uint32 hashes.
        secondary_hashes: (N,) uint32 hashes.
        filled: (N,) bool mask (or scalar bool) for active entries.
        unique_key: Optional cost array; picks min cost per group (ties -> smallest index).
        uint32eds: Optional (N, K) uint32 representation of the values. Required for collision
            detection in safe mode.
    """
    primary_hashes = jnp.asarray(primary_hashes, dtype=jnp.uint32).reshape(-1)
    secondary_hashes = jnp.asarray(secondary_hashes, dtype=jnp.uint32).reshape(-1)
    batch_len = int(primary_hashes.shape[0])
    if secondary_hashes.shape[0] != batch_len:
        raise ValueError("primary_hashes and secondary_hashes must have the same length.")

    # Exact mode semantics: do not rely on hash pairs.
    if _DEDUPE_MODE == "exact":
        if uint32eds is None:
            raise ValueError("exact dedupe requires uint32eds")
        keys = jnp.asarray(uint32eds, dtype=jnp.uint32)
        if keys.ndim == 1:
            keys = keys[:, None]
        return _compute_unique_mask_from_uint32eds(
            uint32eds=keys, filled=filled, unique_key=unique_key
        )

    filled = cast(jax.Array, jnp.asarray(filled, dtype=jnp.bool_))
    if filled.ndim == 0:
        filled = cast(jax.Array, jnp.full((batch_len,), filled, dtype=jnp.bool_))
    elif filled.shape[0] != batch_len:
        raise ValueError("filled must match hash arrays leading dimension.")

    sentinel = jnp.broadcast_to(jnp.uint32(0xFFFFFFFF), primary_hashes.shape)
    h1 = jnp.asarray(jax.lax.select(filled, primary_hashes, sentinel), dtype=jnp.uint32)
    h2 = jnp.asarray(jax.lax.select(filled, secondary_hashes, sentinel), dtype=jnp.uint32)

    indices = jnp.arange(batch_len, dtype=jnp.int32)

    if _SORT_BACKEND == "stable_argsort":

        def _stable_sort_perm(perm_in: jax.Array, key_1d: jax.Array) -> jax.Array:
            order = jnp.argsort(key_1d[perm_in], stable=True)
            return perm_in[order]

        perm = indices
        # Primary key is h1; apply stable sorts from least-significant (h2).
        perm = _stable_sort_perm(perm, h2)
        perm = _stable_sort_perm(perm, h1)

        sorted_indices = cast(jax.Array, perm)
        sorted_h1 = h1[sorted_indices]
        sorted_h2 = h2[sorted_indices]

    else:
        is_stable = _SORT_BACKEND == "lax_stable"
        sorted_h1, sorted_h2, sorted_indices = cast(
            tuple[jax.Array, jax.Array, jax.Array],
            jax.lax.sort(
                (h1, h2, indices),
                dimension=0,
                is_stable=is_stable,
                num_keys=2,
            ),
        )

    sorted_filled = filled[sorted_indices]

    def _compute_from_sorted() -> tuple[chex.Array, chex.Array]:
        if batch_len == 0:
            representative_indices = jnp.zeros((0,), dtype=jnp.int32)
            unique_mask = jnp.zeros((0,), dtype=jnp.bool_)
            return unique_mask, representative_indices

        row_changed = jnp.logical_or(
            sorted_h1[1:] != sorted_h1[:-1],
            sorted_h2[1:] != sorted_h2[:-1],
        )
        is_group_start = jnp.concatenate([jnp.array([True]), row_changed], axis=0)
        group_id = jnp.cumsum(is_group_start.astype(jnp.int32)) - jnp.int32(1)

        batch_len_i32 = jnp.int32(batch_len)

        if unique_key is not None:
            unique_key_arr = cast(jax.Array, jnp.asarray(unique_key))
            if unique_key_arr.ndim == 0:
                unique_key_arr = cast(jax.Array, jnp.full((batch_len,), unique_key_arr))
            elif unique_key_arr.shape[0] != batch_len:
                raise ValueError("unique_key must match hash arrays leading dimension.")

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
            jnp.full((batch_len,), batch_len_i32, dtype=jnp.int32)
            .at[group_id]
            .min(candidate_indices)
        )
        representative_per_group = jnp.where(
            representative_per_group == batch_len_i32,
            jnp.int32(0),
            representative_per_group,
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

        indices_local = jnp.arange(batch_len, dtype=jnp.int32)
        unique_mask = jnp.logical_and(filled, indices_local == representative_indices)
        return unique_mask, cast(jax.Array, representative_indices)

    # Collision detection (safe mode): if two adjacent entries share the same hash pair
    # but have different underlying uint32ed rows, we must fall back to an exact dedupe.
    if _DEDUPE_MODE == "safe":
        if uint32eds is None:
            raise ValueError("safe dedupe requires uint32eds for collision detection")

        keys = jnp.asarray(uint32eds, dtype=jnp.uint32)
        if keys.ndim == 1:
            keys = keys[:, None]
        if keys.shape[0] != batch_len:
            raise ValueError("uint32eds must match hash arrays leading dimension.")

        same_pair = jnp.logical_and(
            sorted_h1[1:] == sorted_h1[:-1],
            sorted_h2[1:] == sorted_h2[:-1],
        )
        same_pair = jnp.logical_and(same_pair, sorted_filled[1:])
        same_pair = jnp.logical_and(same_pair, sorted_filled[:-1])
        has_dups = jnp.any(same_pair)

        def _check_collision(_):
            lhs = keys[sorted_indices[1:]]
            rhs = keys[sorted_indices[:-1]]
            adj_equal = jnp.all(lhs == rhs, axis=1)
            return jnp.any(jnp.logical_and(same_pair, jnp.logical_not(adj_equal)))

        collision = lax.cond(has_dups, _check_collision, lambda _: jnp.bool_(False), operand=None)

        def _fallback(_):
            return _compute_unique_mask_from_uint32eds(
                uint32eds=keys, filled=filled, unique_key=unique_key
            )

        def _no_fallback(_):
            return _compute_from_sorted()

        return lax.cond(collision, _fallback, _no_fallback, operand=None)

    return _compute_from_sorted()


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
    """Calculate a new hash bucket index, probe step, and both hash values from a uint32ed."""
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
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Hash a Xtructurable and return index, step, uint32ed, fingerprint, and hash pair."""
    (primary_hash, secondary_hash), uint32ed = cast(
        tuple[tuple[jax.Array, jax.Array], jax.Array],
        cast(Any, input).hash_pair_with_uint32ed(seed),
    )
    primary_hash = jnp.asarray(primary_hash, dtype=jnp.uint32)
    secondary_hash = jnp.asarray(secondary_hash, dtype=jnp.uint32)
    modulus_u32 = jnp.asarray(modulus, dtype=SIZE_DTYPE)
    modulus_u32 = jnp.maximum(modulus_u32, SIZE_DTYPE(1))
    mask = modulus_u32 - SIZE_DTYPE(1)
    is_pow2 = jnp.logical_and(modulus_u32 > 0, (modulus_u32 & mask) == 0)
    idx = jax.lax.select(
        is_pow2, jnp.asarray(primary_hash, dtype=SIZE_DTYPE) & mask, primary_hash % modulus_u32
    )
    step = _normalize_probe_step(secondary_hash, modulus)
    fingerprint = _mix_fingerprint(primary_hash, secondary_hash, jnp.uint32(0))
    return idx, step, uint32ed, fingerprint, primary_hash, secondary_hash


def get_new_idx_hashed(
    input: Xtructurable,
    modulus: int,
    seed: int,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Hash a Xtructurable and return index, step, fingerprint, and hash pair.

    This avoids materializing/returning the (potentially very wide) uint32ed buffer.
    """
    primary_hash, secondary_hash = cast(
        tuple[jax.Array, jax.Array],
        cast(Any, input).hash_pair(seed),
    )
    primary_hash = jnp.asarray(primary_hash, dtype=jnp.uint32)
    secondary_hash = jnp.asarray(secondary_hash, dtype=jnp.uint32)
    modulus_u32 = jnp.asarray(modulus, dtype=SIZE_DTYPE)
    modulus_u32 = jnp.maximum(modulus_u32, SIZE_DTYPE(1))
    mask = modulus_u32 - SIZE_DTYPE(1)
    is_pow2 = jnp.logical_and(modulus_u32 > 0, (modulus_u32 & mask) == 0)
    idx = jax.lax.select(
        is_pow2, jnp.asarray(primary_hash, dtype=SIZE_DTYPE) & mask, primary_hash % modulus_u32
    )
    step = _normalize_probe_step(secondary_hash, modulus)
    fingerprint = _mix_fingerprint(primary_hash, secondary_hash, jnp.uint32(0))
    return idx, step, fingerprint, primary_hash, secondary_hash
