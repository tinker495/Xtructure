"""Optimized unique_mask implementation using wide hashing and Lexsort."""

from __future__ import annotations

from typing import Any, Callable, Union

import jax
import jax.numpy as jnp
from jax import lax

from ....xtructure_decorators import Xtructurable


def _hash_to_wide(keys: Any) -> list[jax.Array]:
    """Hash (N, K) uint32 keys into a fixed-width wide hash."""
    keys = jnp.asarray(keys)
    n, k = keys.shape

    # Check if x64 is enabled
    use_x64 = jax.config.read("jax_enable_x64")

    # If k is small, no need to hash
    max_direct_cols = 2 if use_x64 else 4
    if k <= max_direct_cols:
        return [keys[:, i] for i in range(k)]

    if use_x64:
        # 128-bit via two uint64
        h1 = keys[:, 0].astype(jnp.uint64)
        h2 = keys[:, 0].astype(jnp.uint64)
        c1 = jnp.uint64(0x9E3779B1)
        c2 = jnp.uint64(0x85EBCA6B)

        def _hash_to_wide_step_u64(i, carry):
            hh1, hh2 = carry
            col = lax.dynamic_index_in_dim(keys, i, axis=1, keepdims=False).astype(jnp.uint64)
            hh1 = hh1 * c1 + col
            hh2 = hh2 * c2 + col
            return hh1, hh2

        h1, h2 = lax.fori_loop(1, k, _hash_to_wide_step_u64, (h1, h2))
        return [h1, h2]
    else:
        # 128-bit via four uint32
        h1 = keys[:, 0].astype(jnp.uint32)
        h2 = keys[:, 0].astype(jnp.uint32)
        h3 = keys[:, 0].astype(jnp.uint32)
        h4 = keys[:, 0].astype(jnp.uint32)
        c1 = jnp.uint32(0x9E3779B1)
        c2 = jnp.uint32(0x85EBCA6B)
        c3 = jnp.uint32(0xC2B2AE35)
        c4 = jnp.uint32(0x278DDE6E)

        def _hash_to_wide_step_u32(i, carry):
            hh1, hh2, hh3, hh4 = carry
            col = lax.dynamic_index_in_dim(keys, i, axis=1, keepdims=False).astype(jnp.uint32)
            hh1 = hh1 * c1 + col
            hh2 = hh2 * c2 + col
            hh3 = jnp.bitwise_xor(hh3, col) * c3
            hh4 = jnp.bitwise_xor(hh4, col) * c4
            return hh1, hh2, hh3, hh4

        h1, h2, h3, h4 = lax.fori_loop(1, k, _hash_to_wide_step_u32, (h1, h2, h3, h4))
        return [h1, h2, h3, h4]


def unique_mask(
    val: Xtructurable,
    key: jnp.ndarray | None = None,
    filled: jnp.ndarray | None = None,
    key_fn: Callable[[Any], jnp.ndarray] | None = None,
    batch_len: int | None = None,
    return_index: bool = False,
    return_inverse: bool = False,
    size: int | None = None,
    fill_value: int | None = None,
) -> Union[jnp.ndarray, tuple]:
    """Mask or index information for selecting unique states.

    Optimized implementation using wide hashing + Lexsort. This approach
    reduces any multi-column key into a fixed-width representation (128-bit),
    minimizing sorting passes and comparison overhead while maintaining
    near-zero collision probability.

    Args:
        val: Xtructurable dataclass to deduplicate.
        key: Optional cost array (e.g. priority). If provided, the item with
             the *lowest* key among duplicates is selected.
        filled: Optional boolean mask indicating valid items. Invalid items
                are treated as non-existent (never selected).
        key_fn: Function to generate hash/comparison keys from `val`.
        batch_len: Explicit batch length (optional).
        return_index: Whether to return indices of unique items.
        return_inverse: Whether to return inverse indices.
        size: Optional static size for returned unique indices (required for JIT).
        fill_value: Value to fill padding with when size is specified.

    Returns:
        Mask (bool array) or tuple (mask, index, inverse).
    """
    key_fn_local = key_fn
    if key_fn_local is None:

        def _key_fn(x):
            # Prefer aggregate-packed representation if available for density
            if hasattr(x, "packed") and hasattr(x, "bitpack_schema"):
                return x.packed.words
            return x.uint32ed

        key_fn_local = _key_fn

    # 1. Generate keys for uniqueness
    try:
        unique_keys = jax.vmap(key_fn_local)(val)
    except Exception as e:
        raise ValueError(f"key_fn failed to generate hashable keys: {e}")

    if batch_len is None:
        batch_len_i = int(val.shape.batch[0])
    else:
        batch_len_i = int(batch_len)

    if batch_len_i == 0:
        final_mask = jnp.zeros((0,), dtype=jnp.bool_)
        if not return_index and not return_inverse:
            return final_mask
        returns = (final_mask,)
        if return_index:
            returns += (jnp.zeros((0,), dtype=jnp.int32),)
        if return_inverse:
            returns += (jnp.zeros((0,), dtype=jnp.int32),)
        return returns

    if key is not None and len(key) != batch_len_i:
        raise ValueError(f"key length {len(key)} must match batch_len {batch_len_i}")

    keys_flat = jnp.asarray(unique_keys).reshape((batch_len_i, -1))

    # 2. Wide Hashing to reduce sort columns (128-bit)
    hashes = _hash_to_wide(keys_flat)

    # 3. Effective mask: filled AND finite(cost) when key is provided.
    if filled is None:
        effective_filled = jnp.ones((batch_len_i,), dtype=jnp.bool_)
    else:
        effective_filled = jnp.asarray(filled, dtype=jnp.bool_)

    if key is not None:
        key_arr = jnp.asarray(key)
        # Docstring semantics: entries with +inf cost are excluded.
        finite_key = key_arr < jnp.array(jnp.inf, dtype=key_arr.dtype)
        effective_filled = jnp.logical_and(effective_filled, finite_key)

    # 4. Handle filled: invalid items get MAX_KEY so they sort to the end
    for i in range(len(hashes)):
        dtype = hashes[i].dtype
        if jnp.issubdtype(dtype, jnp.integer) or dtype == jnp.bool_:
            max_val = jnp.array(jnp.iinfo(dtype).max, dtype=dtype)
        elif jnp.issubdtype(dtype, jnp.floating):
            max_val = jnp.array(jnp.finfo(dtype).max, dtype=dtype)
        else:
            raise TypeError(f"Unsupported key dtype for unique_mask: {dtype}")
        hashes[i] = jnp.where(effective_filled, hashes[i], max_val)

    def _stable_sort_perm(perm_in: jax.Array, key_1d: jax.Array) -> jax.Array:
        order = jnp.argsort(key_1d[perm_in], stable=True)
        return perm_in[order]

    # 5. Stable multi-pass argsort to avoid lexsort / multi-key lax.sort.
    # Start perm as arange to get smallest-index tie-breaking for free.
    perm = jnp.arange(batch_len_i, dtype=jnp.int32)

    if key is not None:
        key_arr2 = jnp.asarray(key)
        inf_fill = jnp.array(jnp.inf, dtype=key_arr2.dtype)
        valid_key = jnp.where(effective_filled, key_arr2, inf_fill)
        perm = _stable_sort_perm(perm, valid_key)

    # Hash keys: least significant first so the first hash becomes most significant.
    for h in reversed(hashes):
        perm = _stable_sort_perm(perm, h)

    # 6. Compute unique mask from sorted hashes
    sorted_hashes = [h[perm] for h in hashes]

    # Row i is unique if it differs from i-1 in ANY of the hashes
    diffs = [h[1:] != h[:-1] for h in sorted_hashes]
    is_diff = diffs[0]
    for d in diffs[1:]:
        is_diff = jnp.logical_or(is_diff, d)

    mask_sorted = jnp.concatenate([jnp.array([True]), is_diff])

    # 7. Filter out the invalid group
    is_valid_sorted = effective_filled[perm]
    mask_sorted = jnp.logical_and(mask_sorted, is_valid_sorted)

    # 8. Map back to original order
    final_mask = jnp.zeros(batch_len_i, dtype=jnp.bool_).at[perm].set(mask_sorted)

    if not return_index and not return_inverse:
        return final_mask

    returns = (final_mask,)

    if return_index:
        use_static = size is not None
        unique_indices = None

        if not use_static:
            try:
                unique_indices = perm[mask_sorted]
            except (jax.errors.NonConcreteBooleanIndexError, jax.errors.ConcretizationTypeError):
                # Fallback to batch_len if concrete
                if isinstance(batch_len_i, (int, jnp.integer)):
                    use_static = True
                    size = int(batch_len_i)
                    if fill_value is None:
                        fill_value = int(batch_len_i)
                else:
                    raise

        if use_static:
            if fill_value is None:
                fill_value = 0

            # JIT-safe path: use nonzero with fixed size
            sentinel = batch_len_i
            valid_positions = jnp.nonzero(mask_sorted, size=size, fill_value=sentinel)[0]

            # Pad perm with fill_value at the sentinel position so invalid lookups get fill_value
            perm_padded = jnp.concatenate([perm, jnp.array([fill_value], dtype=jnp.int32)])
            unique_indices = perm_padded[valid_positions]

        returns += (unique_indices,)

    if return_inverse:
        group_id_sorted = jnp.cumsum(mask_sorted) - 1
        inv = jnp.zeros(batch_len_i, dtype=jnp.int32).at[perm].set(group_id_sorted)
        returns += (inv,)

    return returns
