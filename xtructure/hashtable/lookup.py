"""Lookup helpers for HashTable."""

from __future__ import annotations

from typing import Any, cast

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable
from .constants import SIZE_DTYPE, SLOT_IDX_DTYPE
from .hash_utils import get_new_idx_hashed
from .types import BucketIdx, HashIdx

BucketIdxCls = cast(Any, BucketIdx)
HashIdxCls = cast(Any, HashIdx)


def _hashtable_lookup_internal(
    table: Any,
    input: Xtructurable,
    idx: Any,
    probe_step: chex.Array,
    input_fingerprint: chex.Array,
    found: chex.Array,
) -> tuple[Any, chex.Array]:
    probe_step = jnp.asarray(probe_step, dtype=SIZE_DTYPE)
    bucket_size = int(table.bucket_size)
    max_probes_u32 = SIZE_DTYPE(int(table.max_probes))
    capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)
    mask = capacity - SIZE_DTYPE(1)

    slot_offsets = jnp.arange(bucket_size, dtype=SLOT_IDX_DTYPE)
    slot_offsets_u32 = slot_offsets.astype(SIZE_DTYPE)
    bucket_size_u32 = SIZE_DTYPE(bucket_size)

    def _cond(val: tuple[Any, chex.Array, chex.Array]) -> chex.Array:
        _, found, probes = val
        return jnp.logical_and(jnp.logical_not(found), probes < max_probes_u32)

    def _body(
        val: tuple[Any, chex.Array, chex.Array],
    ) -> tuple[Any, chex.Array, chex.Array]:
        idx, found, probes = val
        bucket_idx = jnp.asarray(idx.index, dtype=SIZE_DTYPE)
        filled_limit = table.bucket_fill_levels[bucket_idx].astype(SLOT_IDX_DTYPE)
        filled_limit_u32 = filled_limit.astype(SIZE_DTYPE)

        flat_indices = bucket_idx * bucket_size_u32 + slot_offsets_u32
        stored_fps = table.fingerprints[flat_indices]

        is_filled = slot_offsets < filled_limit
        fp_matches = jnp.logical_and(is_filled, stored_fps == input_fingerprint)
        has_fp_match = jnp.any(fp_matches)
        first_match_slot = jnp.argmax(fp_matches).astype(SLOT_IDX_DTYPE)
        candidate_flat = bucket_idx * bucket_size_u32 + first_match_slot.astype(
            SIZE_DTYPE
        )

        def _compare_first() -> chex.Array:
            return jnp.asarray(table.table[candidate_flat] == input, dtype=jnp.bool_)

        found_fast = jnp.logical_and(
            has_fp_match,
            jax.lax.cond(has_fp_match, _compare_first, lambda: jnp.bool_(False)),
        )

        other_fp_matches = jnp.logical_and(fp_matches, slot_offsets != first_match_slot)
        need_fallback = jnp.logical_and(
            jnp.logical_not(found_fast), jnp.any(other_fp_matches)
        )

        def _fallback_scan() -> tuple[chex.Array, chex.Array]:
            def _check_slot(off: chex.Array, match: chex.Array) -> chex.Array:
                flat = bucket_idx * bucket_size_u32 + off.astype(SIZE_DTYPE)
                return jax.lax.cond(
                    match,
                    lambda: jnp.asarray(table.table[flat] == input, dtype=jnp.bool_),
                    lambda: jnp.bool_(False),
                )

            slot_equals = jax.vmap(_check_slot)(slot_offsets, other_fp_matches)
            found_in_bucket = jnp.any(slot_equals)
            match_slot = jnp.argmax(slot_equals).astype(SLOT_IDX_DTYPE)
            return found_in_bucket, match_slot

        found_fb, match_fb = jax.lax.cond(
            need_fallback,
            _fallback_scan,
            lambda: (jnp.bool_(False), SLOT_IDX_DTYPE(0)),
        )

        new_found = jnp.logical_or(found_fast, found_fb)
        match_slot = cast(jax.Array, jnp.where(found_fast, first_match_slot, match_fb))

        bucket_full = filled_limit_u32 == bucket_size_u32
        should_stop = jnp.logical_or(new_found, jnp.logical_not(bucket_full))

        out_slot = cast(jax.Array, jnp.where(new_found, match_slot, filled_limit))
        next_bucket = (bucket_idx + probe_step) & mask

        updated_idx = BucketIdxCls(
            index=cast(jax.Array, jnp.where(should_stop, bucket_idx, next_bucket)),
            slot_index=cast(
                jax.Array,
                jnp.where(should_stop, out_slot, SLOT_IDX_DTYPE(0)),
            ),
        )

        probes = probes + bucket_size_u32
        probes = cast(jax.Array, jnp.where(should_stop, max_probes_u32, probes))
        return updated_idx, new_found, probes

    idx, found, _ = jax.lax.while_loop(_cond, _body, (idx, found, jnp.uint32(0)))
    return idx, found


@jax.jit
def _hashtable_lookup_bucket_jit(
    table: Any, input: Xtructurable
) -> tuple[Any, chex.Array, chex.Array]:
    index, step, fingerprint, _primary_hash, _secondary_hash = cast(
        Any,
        get_new_idx_hashed(input, table._capacity, table.seed),
    )
    idx = BucketIdxCls(index=index, slot_index=SLOT_IDX_DTYPE(0))
    idx, found = _hashtable_lookup_internal(
        table, input, idx, step, fingerprint, jnp.bool_(False)
    )
    return idx, found, fingerprint


@jax.jit
def _hashtable_lookup_jit(table: Any, input: Xtructurable) -> tuple[Any, chex.Array]:
    idx, found, _ = _hashtable_lookup_bucket_jit(table, input)
    bucket_size_u32 = SIZE_DTYPE(table.bucket_size)
    return HashIdxCls(index=idx.index * bucket_size_u32 + idx.slot_index), found


def _hashtable_lookup_parallel_internal(
    table: Any,
    inputs: Xtructurable,
    idxs: Any,
    probe_steps: chex.Array,
    fingerprints: chex.Array,
    founds: chex.Array,
    active: chex.Array | None = None,
) -> tuple[Any, chex.Array]:
    if active is None:
        active = jnp.ones(inputs.shape.batch, dtype=jnp.bool_)

    bucket_size = int(table.bucket_size)
    max_probes_u32 = SIZE_DTYPE(int(table.max_probes))
    capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)
    mask = capacity - SIZE_DTYPE(1)
    probe_steps = jnp.asarray(probe_steps, dtype=SIZE_DTYPE)
    fingerprints = cast(jax.Array, jnp.asarray(fingerprints, dtype=jnp.uint32)).reshape(
        (-1,)
    )
    bucket_size_u32 = SIZE_DTYPE(bucket_size)
    slot_offsets = jnp.arange(bucket_size, dtype=SLOT_IDX_DTYPE)
    slot_offsets_u32 = slot_offsets.astype(SIZE_DTYPE)

    probes = jnp.zeros(inputs.shape.batch, dtype=SIZE_DTYPE)

    def _cond(val: tuple[Any, chex.Array, chex.Array, chex.Array]) -> chex.Array:
        _, _, _, active = val
        return jnp.any(active)

    def _body(
        val: tuple[Any, chex.Array, chex.Array, chex.Array],
    ) -> tuple[Any, chex.Array, chex.Array, chex.Array]:
        idxs, founds, probes, active = val

        bucket_indices = jnp.asarray(idxs.index, dtype=SIZE_DTYPE)
        filled_limits = table.bucket_fill_levels[bucket_indices].astype(SLOT_IDX_DTYPE)
        filled_limits_u32 = filled_limits.astype(SIZE_DTYPE)
        under_limit = probes < max_probes_u32

        step_active = jnp.logical_and(
            active,
            jnp.logical_and(jnp.logical_not(founds), under_limit),
        )

        flat_indices = (
            bucket_indices[:, None] * bucket_size_u32 + slot_offsets_u32[None, :]
        )
        stored_fps = table.fingerprints[flat_indices]

        is_filled = slot_offsets[None, :] < filled_limits[:, None]
        fp_matches = jnp.logical_and(
            jnp.logical_and(is_filled, stored_fps == fingerprints[:, None]),
            step_active[:, None],
        )

        has_fp_match_row = jnp.any(fp_matches, axis=1)
        first_match_slot = jnp.argmax(fp_matches, axis=1).astype(SLOT_IDX_DTYPE)
        candidate_flat = bucket_indices * bucket_size_u32 + first_match_slot.astype(
            SIZE_DTYPE
        )

        def _maybe_compare(
            has_match: chex.Array, flat: chex.Array, inp: Xtructurable
        ) -> chex.Array:
            return jax.lax.cond(
                has_match,
                lambda: jnp.asarray(table.table[flat] == inp, dtype=jnp.bool_),
                lambda: jnp.bool_(False),
            )

        found_fast = jax.vmap(_maybe_compare)(
            jnp.logical_and(step_active, has_fp_match_row), candidate_flat, inputs
        )

        other_fp_matches = jnp.logical_and(
            fp_matches,
            slot_offsets[None, :] != first_match_slot[:, None],
        )
        need_fallback = jnp.logical_and(
            jnp.logical_not(found_fast),
            jnp.any(other_fp_matches, axis=1),
        )

        def _fallback_one(
            need: chex.Array,
            flat_row: chex.Array,
            inp: Xtructurable,
            other_row: chex.Array,
        ) -> tuple[chex.Array, chex.Array]:
            def _do():
                bucket_vals = table.table[flat_row]
                value_equals = jax.vmap(lambda s: s == inp)(bucket_vals)
                match_found = jnp.logical_and(value_equals, other_row)
                found_fb = jnp.any(match_found)
                idx_fb = jnp.argmax(match_found).astype(SLOT_IDX_DTYPE)
                return found_fb, idx_fb

            return jax.lax.cond(
                need,
                _do,
                lambda: (jnp.bool_(False), SLOT_IDX_DTYPE(0)),
            )

        found_fb, idx_fb = jax.vmap(_fallback_one)(
            need_fallback, flat_indices, inputs, other_fp_matches
        )

        new_founds_in_bucket = jnp.logical_or(found_fast, found_fb)
        founds = jnp.logical_or(founds, new_founds_in_bucket)

        match_slot = cast(jax.Array, jnp.where(found_fast, first_match_slot, idx_fb))

        bucket_full = filled_limits_u32 == bucket_size_u32

        out_slot = cast(
            jax.Array,
            jnp.where(new_founds_in_bucket, match_slot, filled_limits),
        )

        probes = probes + step_active.astype(SIZE_DTYPE) * bucket_size_u32
        under_limit_next = probes < max_probes_u32

        continue_mask = jnp.logical_and(
            step_active,
            jnp.logical_and(
                jnp.logical_not(new_founds_in_bucket),
                jnp.logical_and(bucket_full, under_limit_next),
            ),
        )

        next_bucket_indices = (bucket_indices + probe_steps) & mask
        updated_idxs = BucketIdxCls(
            index=cast(
                jax.Array,
                jnp.where(continue_mask, next_bucket_indices, bucket_indices),
            ),
            slot_index=cast(
                jax.Array,
                jnp.where(continue_mask, SLOT_IDX_DTYPE(0), out_slot),
            ),
        )
        idxs = BucketIdxCls(
            index=cast(
                jax.Array, jnp.where(step_active, updated_idxs.index, idxs.index)
            ),
            slot_index=cast(
                jax.Array,
                jnp.where(step_active, updated_idxs.slot_index, idxs.slot_index),
            ),
        )

        active = continue_mask
        return idxs, founds, probes, active

    idxs, founds, _, _ = jax.lax.while_loop(
        _cond, _body, (idxs, founds, probes, active)
    )
    return idxs, founds


@jax.jit
def _hashtable_lookup_parallel_jit(
    table: Any, inputs: Xtructurable, filled: chex.Array | bool = True
) -> tuple[Any, chex.Array]:
    filled = jnp.asarray(filled)
    batch_size = inputs.shape.batch

    def _process_batch(filled_mask):
        initial_idx, steps, fingerprints, _primary_hashes, _secondary_hashes = cast(
            Any,
            jax.vmap(get_new_idx_hashed, in_axes=(0, None, None))(
                inputs, table._capacity, table.seed
            ),
        )

        idxs = BucketIdxCls(
            index=initial_idx,
            slot_index=jnp.zeros(batch_size, dtype=SLOT_IDX_DTYPE),
        )
        founds = jnp.zeros(batch_size, dtype=jnp.bool_)

        idx, found = _hashtable_lookup_parallel_internal(
            table,
            inputs,
            idxs,
            steps,
            fingerprints,
            founds,
            filled_mask,
        )
        bucket_size_u32 = SIZE_DTYPE(table.bucket_size)
        return HashIdxCls(index=idx.index * bucket_size_u32 + idx.slot_index), found

    def _empty_result(_):
        return (
            HashIdxCls(index=jnp.zeros(batch_size, dtype=SIZE_DTYPE)),
            jnp.zeros(batch_size, dtype=jnp.bool_),
        )

    if filled.ndim == 0:
        return jax.lax.cond(
            filled,
            lambda: _process_batch(jnp.ones(batch_size, dtype=jnp.bool_)),
            lambda: _empty_result(None),
        )
    else:
        return _process_batch(filled)


@jax.jit
def _hashtable_getitem_jit(table: Any, idx: Any) -> Xtructurable:
    return table.table[idx.index]
