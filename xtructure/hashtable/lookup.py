"""Lookup helpers for HashTable."""

from __future__ import annotations

from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp

from ..core.dtype_facts import SIZE_DTYPE
from ..core.protocol import Xtructurable
from ..core.xtructure_numpy.dataclass_ops.unique_ops import (
    _unique_mask_from_uint32_rows,
)
from .constants import SLOT_IDX_DTYPE
from .hash_utils import (
    _check_probe_matches_inputs,
    get_new_idx_byterized,
    get_new_idx_byterized_batched,
)
from .types import BucketIdx, HashIdx, HashTableProbe


def _hashtable_lookup_internal(
    table: Any,
    input: Xtructurable,
    idx: BucketIdx,
    probe_step: jax.Array,
    input_fingerprint: jax.Array,
    found: bool | jax.Array,
) -> tuple[BucketIdx, jax.Array]:
    found = jnp.asarray(found, dtype=jnp.bool_)
    probe_step = jnp.asarray(probe_step, dtype=SIZE_DTYPE)
    bucket_size = int(table.bucket_size)
    max_probes_u32 = SIZE_DTYPE(int(table.max_probes))
    capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)
    mask = capacity - SIZE_DTYPE(1)

    slot_offsets = jnp.arange(bucket_size, dtype=SLOT_IDX_DTYPE)
    slot_offsets_u32 = slot_offsets.astype(SIZE_DTYPE)
    bucket_size_u32 = SIZE_DTYPE(bucket_size)

    def _cond(val: tuple[BucketIdx, jax.Array, jax.Array]) -> jax.Array:
        _, found, probes = val
        return jnp.logical_and(~found, probes < max_probes_u32)

    def _body(
        val: tuple[BucketIdx, jax.Array, jax.Array]
    ) -> tuple[BucketIdx, jax.Array, jax.Array]:
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
        candidate_flat = bucket_idx * bucket_size_u32 + first_match_slot.astype(SIZE_DTYPE)

        def _compare_first() -> chex.Array:
            return jnp.asarray(table.table[candidate_flat] == input, dtype=jnp.bool_)

        found_fast = jnp.logical_and(
            has_fp_match, jax.lax.cond(has_fp_match, _compare_first, lambda: jnp.bool_(False))
        )

        other_fp_matches = jnp.logical_and(fp_matches, slot_offsets != first_match_slot)
        need_fallback = jnp.logical_and(~found_fast, jnp.any(other_fp_matches))

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
        match_slot = jnp.where(found_fast, first_match_slot, match_fb).astype(SLOT_IDX_DTYPE)

        bucket_full = filled_limit_u32 == bucket_size_u32
        should_stop = jnp.logical_or(new_found, jnp.logical_not(bucket_full))

        out_slot = jnp.where(new_found, match_slot, filled_limit).astype(SLOT_IDX_DTYPE)
        next_bucket = (bucket_idx + probe_step) & mask

        updated_idx = BucketIdx(
            index=jnp.where(should_stop, bucket_idx, next_bucket).astype(SIZE_DTYPE),
            slot_index=jnp.where(should_stop, out_slot, SLOT_IDX_DTYPE(0)),
        )

        probes = probes + bucket_size_u32
        probes = jnp.where(should_stop, max_probes_u32, probes)
        return updated_idx, new_found, probes

    idx, found, _ = jax.lax.while_loop(_cond, _body, (idx, found, jnp.uint32(0)))
    return idx, found


@jax.jit
def _hashtable_lookup_bucket_jit(
    table: Any, input: Xtructurable
) -> tuple[BucketIdx, jax.Array, jax.Array]:
    index, step, _uint32ed, fingerprint = get_new_idx_byterized(input, table._capacity, table.seed)
    idx = BucketIdx(index=index, slot_index=SLOT_IDX_DTYPE(0))
    idx, found = _hashtable_lookup_internal(table, input, idx, step, fingerprint, False)
    return idx, found, fingerprint


@jax.jit
def _hashtable_lookup_jit(table: Any, input: Xtructurable) -> tuple[HashIdx, jax.Array]:
    idx, found, _ = _hashtable_lookup_bucket_jit(table, input)
    bucket_size_u32 = SIZE_DTYPE(table.bucket_size)
    return HashIdx(index=idx.index * bucket_size_u32 + idx.slot_index), found


# Probe rounds run unconditionally before the while_loop. Every while round
# costs a host round-trip (the loop predicate is copied D2H), which dominates
# lookup wall time on GPU; straight-line rounds cost only their (cheap)
# kernels. 2 covers the typical probe depth at benchmark load factors.
_PROLOGUE_PROBE_ROUNDS = 2


def _hashtable_lookup_parallel_internal(
    table: Any,
    inputs: Xtructurable,
    idxs: BucketIdx,
    probe_steps: jax.Array,
    fingerprints: jax.Array,
    founds: jax.Array,
    active: jax.Array | None = None,
) -> tuple[BucketIdx, jax.Array]:
    if active is None:
        active = jnp.ones(inputs.shape, dtype=jnp.bool_)

    bucket_size = int(table.bucket_size)
    max_probes_u32 = SIZE_DTYPE(int(table.max_probes))
    capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)
    mask = capacity - SIZE_DTYPE(1)
    probe_steps = jnp.asarray(probe_steps, dtype=SIZE_DTYPE)
    bucket_size_u32 = SIZE_DTYPE(bucket_size)
    slot_offsets = jnp.arange(bucket_size, dtype=SLOT_IDX_DTYPE)
    slot_offsets_u32 = slot_offsets.astype(SIZE_DTYPE)

    probes = jnp.zeros(inputs.shape, dtype=SIZE_DTYPE)

    def _cond(val: tuple[BucketIdx, jax.Array, jax.Array, jax.Array]) -> jax.Array:
        _, _, _, active = val
        return jnp.any(active)

    def _body(
        val: tuple[BucketIdx, jax.Array, jax.Array, jax.Array],
    ) -> tuple[BucketIdx, jax.Array, jax.Array, jax.Array]:
        idxs, founds, probes, active = val

        bucket_indices = jnp.asarray(idxs.index, dtype=SIZE_DTYPE)
        filled_limits = table.bucket_fill_levels[bucket_indices].astype(SLOT_IDX_DTYPE)
        filled_limits_u32 = filled_limits.astype(SIZE_DTYPE)
        under_limit = probes < max_probes_u32

        step_active = jnp.logical_and(active, jnp.logical_and(~founds, under_limit))

        flat_indices = bucket_indices[:, None] * bucket_size_u32 + slot_offsets_u32[None, :]
        stored_fps = table.fingerprints[flat_indices]

        is_filled = slot_offsets[None, :] < filled_limits[:, None]
        fp_matches = jnp.logical_and(
            jnp.logical_and(is_filled, stored_fps == fingerprints[:, None]),
            step_active[:, None],
        )

        has_fp_match_row = jnp.any(fp_matches, axis=1)
        first_match_slot = jnp.argmax(fp_matches, axis=1).astype(SLOT_IDX_DTYPE)
        candidate_flat = bucket_indices * bucket_size_u32 + first_match_slot.astype(SIZE_DTYPE)

        def _maybe_compare(has_match: jax.Array, flat: jax.Array, inp: Xtructurable) -> jax.Array:
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
        need_fallback = jnp.logical_and(~found_fast, jnp.any(other_fp_matches, axis=1))

        # Unconditional: when need_fallback is all-False the masking below
        # yields exactly (all-False, 0) — same as the old lax.cond false
        # branch — and skipping the cond avoids a per-round D2H sync of its
        # scalar predicate, which costs far more than this gather.
        bucket_vals = table.table[flat_indices]
        value_equals = jax.vmap(lambda bv, inp: jax.vmap(lambda s: s == inp)(bv))(
            bucket_vals, inputs
        )
        match_found = jnp.logical_and(value_equals, other_fp_matches)
        match_found = jnp.logical_and(match_found, need_fallback[:, None])
        found_fb = jnp.any(match_found, axis=1)
        idx_fb = jnp.argmax(match_found, axis=1).astype(SLOT_IDX_DTYPE)

        new_founds_in_bucket = jnp.logical_or(found_fast, found_fb)
        founds = jnp.logical_or(founds, new_founds_in_bucket)

        match_slot = jnp.where(found_fast, first_match_slot, idx_fb).astype(SLOT_IDX_DTYPE)

        bucket_full = filled_limits_u32 == bucket_size_u32

        out_slot = jnp.where(new_founds_in_bucket, match_slot, filled_limits).astype(SLOT_IDX_DTYPE)

        probes = probes + step_active.astype(SIZE_DTYPE) * bucket_size_u32
        under_limit_next = probes < max_probes_u32

        continue_mask = jnp.logical_and(
            step_active,
            jnp.logical_and(~new_founds_in_bucket, jnp.logical_and(bucket_full, under_limit_next)),
        )

        next_bucket_indices = (bucket_indices + probe_steps) & mask
        updated_idxs = BucketIdx(
            index=jnp.where(continue_mask, next_bucket_indices, bucket_indices).astype(SIZE_DTYPE),
            slot_index=jnp.where(continue_mask, SLOT_IDX_DTYPE(0), out_slot),
        )
        idxs = BucketIdx(
            index=jnp.where(step_active, updated_idxs.index, idxs.index),
            slot_index=jnp.where(step_active, updated_idxs.slot_index, idxs.slot_index),
        )

        active = continue_mask
        return idxs, founds, probes, active

    # A round with active all-False leaves the carry bit-identical (every
    # update is masked by step_active/continue_mask), so unconditionally
    # unrolling the first rounds preserves the while_loop semantics exactly
    # while skipping their host-synced loop-predicate checks.
    val = (idxs, founds, probes, active)
    for _ in range(_PROLOGUE_PROBE_ROUNDS):
        val = _body(val)
    idxs, founds, _, _ = jax.lax.while_loop(_cond, _body, val)
    return idxs, founds


def _empty_lookup_result(
    batch_shape: tuple[int, ...],
) -> tuple[HashIdx, jax.Array]:
    return (
        HashIdx(index=jnp.zeros(batch_shape, dtype=SIZE_DTYPE)),
        jnp.zeros(batch_shape, dtype=jnp.bool_),
    )


def _validate_batch_and_probe(
    inputs: Xtructurable,
    probe: HashTableProbe | None,
) -> tuple[tuple[int, ...], int]:
    batch_shape = inputs.shape
    if len(batch_shape) != 1:
        raise ValueError(f"lookup_parallel expects a 1D batch, got shape {batch_shape}")
    if probe is not None:
        _check_probe_matches_inputs(probe, inputs)
    return batch_shape, batch_shape[0]


def _resolve_probe(
    table: Any,
    inputs: Xtructurable,
    probe: HashTableProbe | None,
) -> HashTableProbe:
    if probe is not None:
        _check_probe_matches_inputs(probe, inputs)
        return probe
    return get_new_idx_byterized_batched(inputs, table._capacity, table.seed)


def _dedup_uint32_rows(
    uint32ed: jax.Array,
    filled_mask: jax.Array | None,
    batch_len: int,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(unique_mask, representative_indices)`` without device-to-host syncs."""
    unique_mask, perm, mask_sorted = _unique_mask_from_uint32_rows(
        keys_flat=uint32ed,
        key=None,
        filled=filled_mask,
        batch_len=batch_len,
    )
    iota = jnp.arange(batch_len, dtype=jnp.int32)
    winner_sorted_pos = jax.lax.cummax(jnp.where(mask_sorted, iota, 0), axis=0)
    rep_sorted = perm[winner_sorted_pos].astype(jnp.int32)
    if filled_mask is not None:
        rep_sorted = jnp.where(filled_mask[perm], rep_sorted, jnp.int32(0))
    representative_indices = jnp.zeros((batch_len,), dtype=jnp.int32).at[perm].set(rep_sorted)
    return unique_mask, representative_indices


def _lookup_parallel_from_probe(
    table: Any,
    inputs: Xtructurable,
    probe: HashTableProbe,
    filled: chex.Array | None,
    *,
    assume_unique: bool = False,
) -> tuple[HashIdx, jax.Array]:
    """Run deduplicated parallel lookup from a precomputed :class:`HashTableProbe`.

    Probes the table at most once per distinct active state and gathers the
    representative ``(HashIdx, found)`` back to every active duplicate row.
    """
    batch_shape, batch_len = _validate_batch_and_probe(inputs, None)
    if batch_len == 0:
        if filled is not None:
            filled_arr = jnp.asarray(filled, dtype=jnp.bool_)
            if filled_arr.ndim not in (0, 1) or (
                filled_arr.ndim == 1 and filled_arr.shape != batch_shape
            ):
                raise ValueError(
                    f"filled shape {filled_arr.shape} does not match batch shape {batch_shape}"
                )
        return _empty_lookup_result(batch_shape)

    bucket_size_u32 = SIZE_DTYPE(table.bucket_size)

    def _process_batch(
        filled_mask: jax.Array | None,
    ) -> tuple[HashIdx, jax.Array]:
        if assume_unique:
            active_mask = (
                jnp.ones(batch_shape, dtype=jnp.bool_) if filled_mask is None else filled_mask
            )
            representative_indices = None
        else:
            active_mask, representative_indices = _dedup_uint32_rows(
                probe.uint32ed,
                filled_mask,
                batch_len,
            )

        idxs = BucketIdx(
            index=probe.index,
            slot_index=jnp.zeros(batch_shape, dtype=SLOT_IDX_DTYPE),
        )
        founds = jnp.zeros(batch_shape, dtype=jnp.bool_)
        probed_idx, probed_found = _hashtable_lookup_parallel_internal(
            table,
            inputs,
            idxs,
            probe.step,
            probe.fingerprint,
            founds,
            active_mask,
        )
        flat_idx = probed_idx.index * bucket_size_u32 + probed_idx.slot_index

        if representative_indices is not None:
            flat_idx = flat_idx[representative_indices]
            probed_found = probed_found[representative_indices]

        if filled_mask is not None:
            flat_idx = jnp.where(filled_mask, flat_idx, SIZE_DTYPE(0))
            probed_found = jnp.logical_and(probed_found, filled_mask)

        return HashIdx(index=flat_idx), probed_found

    if filled is None:
        return _process_batch(None)

    filled_arr = jnp.asarray(filled, dtype=jnp.bool_)
    if filled_arr.ndim == 0:
        return jax.lax.cond(
            filled_arr,
            lambda: _process_batch(None),
            lambda: _empty_lookup_result(batch_shape),
        )
    if filled_arr.ndim != 1 or filled_arr.shape != batch_shape:
        raise ValueError(
            f"filled shape {filled_arr.shape} does not match batch shape {batch_shape}"
        )
    return _process_batch(filled_arr)


@partial(jax.jit, static_argnames=("assume_unique",))
def _hashtable_lookup_parallel_all_jit(
    table: Any,
    inputs: Xtructurable,
    probe: HashTableProbe | None = None,
    assume_unique: bool = False,
) -> tuple[HashIdx, jax.Array]:
    local_probe = _resolve_probe(table, inputs, probe)
    return _lookup_parallel_from_probe(
        table,
        inputs,
        local_probe,
        None,
        assume_unique=assume_unique,
    )


@partial(jax.jit, static_argnames=("assume_unique",))
def _hashtable_lookup_parallel_jit(
    table: Any,
    inputs: Xtructurable,
    filled: chex.Array,
    probe: HashTableProbe | None = None,
    assume_unique: bool = False,
) -> tuple[HashIdx, jax.Array]:
    local_probe = _resolve_probe(table, inputs, probe)
    return _lookup_parallel_from_probe(
        table,
        inputs,
        local_probe,
        filled,
        assume_unique=assume_unique,
    )


def _lookup_parallel_dispatch(
    table: Any,
    inputs: Xtructurable,
    filled: chex.Array | bool | None = True,
    probe: HashTableProbe | None = None,
    *,
    assume_unique: bool = False,
) -> tuple[HashIdx, jax.Array]:
    """Route Python-bool/None ``filled`` to a JIT with no scalar mask operand."""
    if filled is None or isinstance(filled, bool):
        if filled is None or filled:
            return _hashtable_lookup_parallel_all_jit(
                table,
                inputs,
                probe,
                assume_unique=assume_unique,
            )
        batch_shape, _ = _validate_batch_and_probe(inputs, probe)
        return _empty_lookup_result(batch_shape)
    return _hashtable_lookup_parallel_jit(
        table,
        inputs,
        filled,
        probe,
        assume_unique=assume_unique,
    )


@partial(jax.jit, static_argnames=("assume_unique",))
def _hashtable_lookup_parallel_with_probe_all_jit(
    table: Any,
    inputs: Xtructurable,
    probe: HashTableProbe | None = None,
    assume_unique: bool = False,
) -> tuple[HashIdx, jax.Array, HashTableProbe]:
    local_probe = _resolve_probe(table, inputs, probe)
    idx, found = _lookup_parallel_from_probe(
        table,
        inputs,
        local_probe,
        None,
        assume_unique=assume_unique,
    )
    return idx, found, local_probe


@jax.jit
def _hashtable_lookup_parallel_with_probe_none_jit(
    table: Any,
    inputs: Xtructurable,
    probe: HashTableProbe | None = None,
) -> tuple[HashIdx, jax.Array, HashTableProbe]:
    local_probe = _resolve_probe(table, inputs, probe)
    idx, found = _empty_lookup_result(inputs.shape)
    return idx, found, local_probe


@partial(jax.jit, static_argnames=("assume_unique",))
def _hashtable_lookup_parallel_with_probe_jit(
    table: Any,
    inputs: Xtructurable,
    filled: chex.Array,
    probe: HashTableProbe | None = None,
    assume_unique: bool = False,
) -> tuple[HashIdx, jax.Array, HashTableProbe]:
    local_probe = _resolve_probe(table, inputs, probe)
    idx, found = _lookup_parallel_from_probe(
        table,
        inputs,
        local_probe,
        filled,
        assume_unique=assume_unique,
    )
    return idx, found, local_probe


def _lookup_parallel_with_probe_dispatch(
    table: Any,
    inputs: Xtructurable,
    filled: chex.Array | bool | None = True,
    probe: HashTableProbe | None = None,
    *,
    assume_unique: bool = False,
) -> tuple[HashIdx, jax.Array, HashTableProbe]:
    """Route Python-bool/None ``filled`` to a JIT with no scalar mask operand."""
    if filled is None or isinstance(filled, bool):
        if filled is None or filled:
            return _hashtable_lookup_parallel_with_probe_all_jit(
                table,
                inputs,
                probe,
                assume_unique=assume_unique,
            )
        _validate_batch_and_probe(inputs, probe)
        return _hashtable_lookup_parallel_with_probe_none_jit(table, inputs, probe)
    return _hashtable_lookup_parallel_with_probe_jit(
        table,
        inputs,
        filled,
        probe,
        assume_unique=assume_unique,
    )


@jax.jit
def _hashtable_getitem_jit(table: Any, idx: HashIdx) -> Xtructurable:
    return table.table[idx.index]
