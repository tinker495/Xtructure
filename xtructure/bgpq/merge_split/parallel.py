import os
from functools import lru_cache
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pl_triton

from ...core.protocol import Xtructurable
from .common import binary_search_partition

_AUTO_BLOCK_THRESHOLDS = (2**15,)
_AUTO_BLOCK_SIZES = (16, 64)
_DEFAULT_UNROLL_MAX = 32
_DEFAULT_VALUE_PACKING = "auto"
_DEFAULT_VALUE_SCALAR_MAX = 16


def _parse_block_size_override() -> int | None:
    env_value = os.environ.get("XTRUCTURE_BGPQ_MERGE_BLOCK_SIZE")
    if env_value is None:
        return None

    if env_value.strip().lower() == "auto":
        return None

    try:
        block_size = int(env_value)
    except ValueError as exc:
        raise ValueError("XTRUCTURE_BGPQ_MERGE_BLOCK_SIZE must be an integer.") from exc

    if block_size <= 0:
        raise ValueError("XTRUCTURE_BGPQ_MERGE_BLOCK_SIZE must be positive.")
    return block_size


def _parse_unroll_max() -> int:
    env_value = os.environ.get("XTRUCTURE_BGPQ_MERGE_UNROLL_MAX")
    if env_value is None:
        return _DEFAULT_UNROLL_MAX
    try:
        unroll_max = int(env_value)
    except ValueError as exc:
        raise ValueError("XTRUCTURE_BGPQ_MERGE_UNROLL_MAX must be an integer.") from exc
    if unroll_max < 0:
        raise ValueError("XTRUCTURE_BGPQ_MERGE_UNROLL_MAX must be non-negative.")
    return unroll_max


def _parse_triton_param(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    if value.strip().lower() in {"auto", "none", ""}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _parse_value_packing() -> str:
    value = os.environ.get("XTRUCTURE_BGPQ_MERGE_VALUE_PACKING", _DEFAULT_VALUE_PACKING)
    value = value.strip().lower()
    if value not in {"auto", "pad", "scalar", "shard"}:
        raise ValueError(
            "XTRUCTURE_BGPQ_MERGE_VALUE_PACKING must be one of: auto, pad, scalar, shard."
        )
    return value


def _parse_value_scalar_max() -> int:
    value = os.environ.get("XTRUCTURE_BGPQ_MERGE_VALUE_SCALAR_MAX", str(_DEFAULT_VALUE_SCALAR_MAX))
    try:
        scalar_max = int(value)
    except ValueError as exc:
        raise ValueError("XTRUCTURE_BGPQ_MERGE_VALUE_SCALAR_MAX must be an integer.") from exc
    if scalar_max <= 0:
        raise ValueError("XTRUCTURE_BGPQ_MERGE_VALUE_SCALAR_MAX must be positive.")
    return scalar_max


def _select_block_size(total_len: int) -> int:
    override = _parse_block_size_override()
    if override is not None:
        return override

    if jax.default_backend() != "gpu":
        return _AUTO_BLOCK_SIZES[-1]

    for threshold, block_size in zip(_AUTO_BLOCK_THRESHOLDS, _AUTO_BLOCK_SIZES):
        if total_len <= threshold:
            return block_size
    return _AUTO_BLOCK_SIZES[-1]


def _validate_block_size(block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    return block_size


def _make_merge_parallel_kernel(block_size: int, unroll_max: int):
    def merge_parallel_kernel(ak_ref, bk_ref, orig_n_ref, merged_keys_ref, merged_indices_ref):
        """
        Pallas kernel that merges two sorted arrays in parallel using the
        Merge Path algorithm for block-level partitioning.
        """
        block_idx = pl.program_id(axis=0)

        n, m = ak_ref.shape[0], bk_ref.shape[0]

        k_start = block_idx * block_size
        pl.multiple_of(k_start, block_size)

        a_start, b_start = binary_search_partition(k_start, ak_ref, bk_ref)

        idx_dtype = merged_indices_ref.dtype
        idx_a = jnp.asarray(a_start, dtype=idx_dtype)
        idx_b = jnp.asarray(b_start, dtype=idx_dtype)
        orig_n = jnp.asarray(orig_n_ref[()], dtype=idx_dtype)

        def loop_body(step, current_idx_a, current_idx_b):
            out_ptr = k_start + step

            safe_idx_a = jnp.minimum(current_idx_a, n - 1)
            safe_idx_b = jnp.minimum(current_idx_b, m - 1)

            val_a = ak_ref[safe_idx_a]
            val_b = bk_ref[safe_idx_b]

            a_exhausted = current_idx_a >= n
            b_exhausted = current_idx_b >= m
            take_a = jnp.logical_or(b_exhausted, jnp.logical_and(~a_exhausted, val_a <= val_b))

            merged_keys_ref[out_ptr] = jax.lax.select(take_a, val_a, val_b)
            merged_indices_ref[out_ptr] = jax.lax.select(
                take_a,
                current_idx_a,
                orig_n + current_idx_b,
            )

            take_a_i = take_a.astype(idx_dtype)
            take_b_i = jnp.logical_not(take_a).astype(idx_dtype)
            return current_idx_a + take_a_i, current_idx_b + take_b_i

        if unroll_max and block_size <= unroll_max:
            current_idx_a = idx_a
            current_idx_b = idx_b
            for step in range(block_size):
                current_idx_a, current_idx_b = loop_body(step, current_idx_a, current_idx_b)
        else:

            def fori_body(step, state):
                current_idx_a, current_idx_b = state
                return loop_body(step, current_idx_a, current_idx_b)

            jax.lax.fori_loop(0, block_size, fori_body, (idx_a, idx_b))

    return merge_parallel_kernel


def _make_merge_parallel_kernel_kv(block_size: int, unroll_max: int, leaf_count: int):
    def merge_parallel_kernel(ak_ref, bk_ref, *refs):
        block_idx = pl.program_id(axis=0)

        n, m = ak_ref.shape[0], bk_ref.shape[0]

        k_start = block_idx * block_size
        pl.multiple_of(k_start, block_size)

        a_start, b_start = binary_search_partition(k_start, ak_ref, bk_ref)

        idx_dtype = jnp.int32
        idx_a = jnp.asarray(a_start, dtype=idx_dtype)
        idx_b = jnp.asarray(b_start, dtype=idx_dtype)

        av_refs = refs[:leaf_count]
        bv_refs = refs[leaf_count : 2 * leaf_count]
        out_keys_ref = refs[2 * leaf_count]
        out_val_refs = refs[2 * leaf_count + 1 :]

        def loop_body(step, current_idx_a, current_idx_b):
            out_ptr = k_start + step

            safe_idx_a = jnp.minimum(current_idx_a, n - 1)
            safe_idx_b = jnp.minimum(current_idx_b, m - 1)

            val_a = ak_ref[safe_idx_a]
            val_b = bk_ref[safe_idx_b]

            a_exhausted = current_idx_a >= n
            b_exhausted = current_idx_b >= m
            take_a = jnp.logical_or(b_exhausted, jnp.logical_and(~a_exhausted, val_a <= val_b))

            out_keys_ref[out_ptr] = jax.lax.select(take_a, val_a, val_b)

            for av_ref, bv_ref, out_ref in zip(av_refs, bv_refs, out_val_refs):
                val_a_leaf = av_ref[pl.dslice(safe_idx_a, 1)]
                val_b_leaf = bv_ref[pl.dslice(safe_idx_b, 1)]
                out_ref[pl.dslice(out_ptr, 1)] = jax.lax.select(take_a, val_a_leaf, val_b_leaf)

            take_a_i = take_a.astype(idx_dtype)
            take_b_i = jnp.logical_not(take_a).astype(idx_dtype)
            return current_idx_a + take_a_i, current_idx_b + take_b_i

        if unroll_max and block_size <= unroll_max:
            current_idx_a = idx_a
            current_idx_b = idx_b
            for step in range(block_size):
                current_idx_a, current_idx_b = loop_body(step, current_idx_a, current_idx_b)
        else:

            def fori_body(step, state):
                current_idx_a, current_idx_b = state
                return loop_body(step, current_idx_a, current_idx_b)

            jax.lax.fori_loop(0, block_size, fori_body, (idx_a, idx_b))

    return merge_parallel_kernel


@lru_cache(maxsize=None)
def _get_merge_arrays_parallel(
    block_size: int,
    unroll_max: int,
    num_warps: int | None,
    num_stages: int | None,
):
    block_size = _validate_block_size(block_size)
    merge_parallel_kernel = _make_merge_parallel_kernel(block_size, unroll_max)
    compiler_params = None
    if num_warps is not None or num_stages is not None:
        compiler_params = pl_triton.CompilerParams(num_warps=num_warps, num_stages=num_stages)

    @jax.jit
    def _merge_arrays_parallel(ak: jax.Array, bk: jax.Array) -> Tuple[jax.Array, jax.Array]:
        if ak.ndim != 1 or bk.ndim != 1:
            raise ValueError("Input arrays ak and bk must be 1D.")

        n, m = ak.shape[0], bk.shape[0]
        total_len = n + m
        if total_len == 0:
            key_dtype = jnp.result_type(ak.dtype, bk.dtype)
            return jnp.array([], dtype=key_dtype), jnp.array([], dtype=jnp.int32)

        if n == 0:
            key_dtype = jnp.result_type(ak.dtype, bk.dtype)
            return bk.astype(key_dtype), jnp.arange(m, dtype=jnp.int32)

        if m == 0:
            key_dtype = jnp.result_type(ak.dtype, bk.dtype)
            return ak.astype(key_dtype), jnp.arange(n, dtype=jnp.int32)

        key_dtype = jnp.result_type(ak.dtype, bk.dtype)
        ak = ak.astype(key_dtype)
        bk = bk.astype(key_dtype)

        total_len_padded = ((total_len + block_size - 1) // block_size) * block_size
        out_keys_shape_dtype = jax.ShapeDtypeStruct((total_len_padded,), key_dtype)
        out_idx_shape_dtype = jax.ShapeDtypeStruct((total_len_padded,), jnp.int32)

        grid_size = total_len_padded // block_size

        sorted_key_full, sorted_idx_full = pl.pallas_call(
            merge_parallel_kernel,
            grid=(grid_size,),
            out_shape=(out_keys_shape_dtype, out_idx_shape_dtype),
            backend="triton",
            compiler_params=compiler_params,
        )(
            ak,
            bk,
            jnp.array(n, dtype=jnp.int32),
        )
        return sorted_key_full[:total_len], sorted_idx_full[:total_len]

    return _merge_arrays_parallel


@lru_cache(maxsize=None)
def _get_merge_arrays_parallel_kv(
    block_size: int,
    unroll_max: int,
    num_warps: int | None,
    num_stages: int | None,
    leaf_count: int,
):
    block_size = _validate_block_size(block_size)
    merge_parallel_kernel = _make_merge_parallel_kernel_kv(block_size, unroll_max, leaf_count)
    compiler_params = None
    if num_warps is not None or num_stages is not None:
        compiler_params = pl_triton.CompilerParams(num_warps=num_warps, num_stages=num_stages)

    def _merge_arrays_parallel_kv(ak, bk, av_leaves, bv_leaves):
        n, m = ak.shape[0], bk.shape[0]
        total_len = n + m
        if total_len == 0:
            key_dtype = jnp.result_type(ak.dtype, bk.dtype)
            out_keys = jnp.array([], dtype=key_dtype)
            out_vals = [jnp.empty((0,) + leaf.shape[1:], dtype=leaf.dtype) for leaf in av_leaves]
            return out_keys, out_vals

        key_dtype = jnp.result_type(ak.dtype, bk.dtype)
        ak = ak.astype(key_dtype)
        bk = bk.astype(key_dtype)

        total_len_padded = ((total_len + block_size - 1) // block_size) * block_size
        out_keys_shape_dtype = jax.ShapeDtypeStruct((total_len_padded,), key_dtype)
        out_val_shapes = [
            jax.ShapeDtypeStruct((total_len_padded,) + leaf.shape[1:], leaf.dtype)
            for leaf in av_leaves
        ]
        out_shape = (out_keys_shape_dtype, *out_val_shapes)

        grid_size = total_len_padded // block_size

        inputs = (ak, bk, *av_leaves, *bv_leaves)
        outputs = pl.pallas_call(
            merge_parallel_kernel,
            grid=(grid_size,),
            out_shape=out_shape,
            backend="triton",
            compiler_params=compiler_params,
        )(*inputs)

        out_keys = outputs[0][:total_len]
        out_vals_flat = [out[:total_len] for out in outputs[1:]]
        return out_keys, out_vals_flat

    return _merge_arrays_parallel_kv


def merge_arrays_parallel(ak: chex.Array, bk: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """
    Merges two sorted JAX arrays using the parallel Merge Path Pallas kernel.
    """
    if ak.ndim != 1 or bk.ndim != 1:
        raise ValueError("Input arrays ak and bk must be 1D.")

    block_size = _select_block_size(ak.shape[0] + bk.shape[0])
    unroll_max = _parse_unroll_max()
    num_warps = _parse_triton_param("XTRUCTURE_BGPQ_MERGE_NUM_WARPS")
    num_stages = _parse_triton_param("XTRUCTURE_BGPQ_MERGE_NUM_STAGES")
    return _get_merge_arrays_parallel(block_size, unroll_max, num_warps, num_stages)(ak, bk)


def merge_arrays_parallel_kv(
    ak: chex.Array, av: Xtructurable, bk: chex.Array, bv: Xtructurable
) -> Tuple[chex.Array, Xtructurable]:
    if jax.default_backend() != "gpu":
        raise ValueError("merge_arrays_parallel_kv requires a GPU backend.")

    if ak.ndim != 1 or bk.ndim != 1:
        raise ValueError("Input arrays ak and bk must be 1D.")

    av_leaves, treedef = jax.tree_util.tree_flatten(av)
    bv_leaves, treedef_b = jax.tree_util.tree_flatten(bv)
    if treedef != treedef_b:
        raise ValueError("Value trees for av/bv must have matching structure.")

    packed_av_leaves = []
    packed_bv_leaves = []
    pack_specs = []

    packing_mode = _parse_value_packing()
    scalar_max = _parse_value_scalar_max()

    def _next_power_of_two(value: int) -> int:
        if value <= 1:
            return 1
        return 1 << (value - 1).bit_length()

    def _is_power_of_two(value: int) -> bool:
        return value > 0 and (value & (value - 1) == 0)

    for av_leaf, bv_leaf in zip(av_leaves, bv_leaves):
        if av_leaf.shape[1:] != bv_leaf.shape[1:]:
            raise ValueError("Value leaves for av/bv must have matching inner shapes.")
        if av_leaf.shape[0] != ak.shape[0] or bv_leaf.shape[0] != bk.shape[0]:
            raise ValueError("All value leaves must align with key length.")

        inner_shape = av_leaf.shape[1:]
        inner_size = 1
        for dim in inner_shape:
            inner_size *= dim

        use_scalar = False
        use_shard = False
        if packing_mode == "scalar":
            use_scalar = True
        elif packing_mode == "shard":
            use_shard = True
        elif packing_mode == "auto":
            if not _is_power_of_two(inner_size):
                use_shard = inner_size <= scalar_max

        av_flat = av_leaf.reshape((av_leaf.shape[0], inner_size))
        bv_flat = bv_leaf.reshape((bv_leaf.shape[0], inner_size))

        if use_scalar:
            for idx in range(inner_size):
                packed_av_leaves.append(av_flat[:, idx])
                packed_bv_leaves.append(bv_flat[:, idx])
            pack_specs.append(("scalar", inner_shape, inner_size, inner_size))
            continue

        if use_shard:
            shard_sizes = []
            remaining = inner_size
            while remaining > 0:
                shard = 1 << (remaining.bit_length() - 1)
                shard_sizes.append(shard)
                remaining -= shard

            start = 0
            for shard in shard_sizes:
                packed_av_leaves.append(av_flat[:, start : start + shard])
                packed_bv_leaves.append(bv_flat[:, start : start + shard])
                start += shard
            pack_specs.append(("shard", inner_shape, inner_size, shard_sizes))
            continue

        padded_size = _next_power_of_two(inner_size)
        if padded_size != inner_size:
            pad_width = ((0, 0), (0, padded_size - inner_size))
            av_flat = jnp.pad(av_flat, pad_width, mode="constant")
            bv_flat = jnp.pad(bv_flat, pad_width, mode="constant")

        packed_av_leaves.append(av_flat)
        packed_bv_leaves.append(bv_flat)
        pack_specs.append(("pad", inner_shape, inner_size, padded_size))

    block_size = _select_block_size(ak.shape[0] + bk.shape[0])
    unroll_max = _parse_unroll_max()
    num_warps = _parse_triton_param("XTRUCTURE_BGPQ_MERGE_NUM_WARPS")
    num_stages = _parse_triton_param("XTRUCTURE_BGPQ_MERGE_NUM_STAGES")

    merge_kv = _get_merge_arrays_parallel_kv(
        block_size, unroll_max, num_warps, num_stages, len(packed_av_leaves)
    )
    merged_keys, out_vals_flat = merge_kv(ak, bk, packed_av_leaves, packed_bv_leaves)

    restored_leaves = []
    offset = 0
    for mode, inner_shape, inner_size, padding_info in pack_specs:
        if mode == "scalar":
            slice_vals = out_vals_flat[offset : offset + inner_size]
            offset += inner_size
            stacked = jnp.stack(slice_vals, axis=1)
            restored_leaves.append(stacked.reshape((stacked.shape[0],) + inner_shape))
        elif mode == "shard":
            shard_sizes = padding_info
            shard_count = len(shard_sizes)
            slice_vals = out_vals_flat[offset : offset + shard_count]
            offset += shard_count
            concatenated = jnp.concatenate(slice_vals, axis=1)
            restored_leaves.append(concatenated.reshape((concatenated.shape[0],) + inner_shape))
        else:
            leaf = out_vals_flat[offset]
            offset += 1
            trimmed = leaf[:, :inner_size] if padding_info != inner_size else leaf
            restored_leaves.append(trimmed.reshape((trimmed.shape[0],) + inner_shape))

    restored_vals = jax.tree_util.tree_unflatten(treedef, restored_leaves)
    return merged_keys, restored_vals


def _merge_parallel_config(total_len: int) -> Dict[str, Any]:
    return {
        "block_size": _select_block_size(total_len),
        "unroll_max": _parse_unroll_max(),
        "num_warps": _parse_triton_param("XTRUCTURE_BGPQ_MERGE_NUM_WARPS"),
        "num_stages": _parse_triton_param("XTRUCTURE_BGPQ_MERGE_NUM_STAGES"),
        "value_packing": _parse_value_packing(),
        "value_scalar_max": _parse_value_scalar_max(),
    }
