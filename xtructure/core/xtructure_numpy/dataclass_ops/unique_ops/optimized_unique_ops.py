"""Optimized unique_mask implementation using wide hashing and Lexsort."""

from __future__ import annotations

from typing import Any, Callable, Union

import jax
import jax.numpy as jnp

from ....xtructure_decorators import Xtructurable


def _hash_to_wide(keys: jnp.ndarray) -> list[jnp.ndarray]:
    """Hash (N, K) uint32 keys into a fixed-width wide hash."""
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
        for i in range(1, k):
            h1 = h1 * jnp.uint64(0x9E3779B1) + keys[:, i].astype(jnp.uint64)
            h2 = h2 * jnp.uint64(0x85EBCA6B) + keys[:, i].astype(jnp.uint64)
        return [h1, h2]
    else:
        # 128-bit via four uint32
        h1 = keys[:, 0].astype(jnp.uint32)
        h2 = keys[:, 0].astype(jnp.uint32)
        h3 = keys[:, 0].astype(jnp.uint32)
        h4 = keys[:, 0].astype(jnp.uint32)
        for i in range(1, k):
            h1 = h1 * jnp.uint32(0x9E3779B1) + keys[:, i].astype(jnp.uint32)
            h2 = h2 * jnp.uint32(0x85EBCA6B) + keys[:, i].astype(jnp.uint32)
            h3 = jnp.bitwise_xor(h3, keys[:, i].astype(jnp.uint32)) * jnp.uint32(0xC2B2AE35)
            h4 = jnp.bitwise_xor(h4, keys[:, i].astype(jnp.uint32)) * jnp.uint32(0x278DDE6E)
        return [h1, h2, h3, h4]


def unique_mask(
    val: Xtructurable,
    key: jnp.ndarray | None = None,
    filled: jnp.ndarray | None = None,
    key_fn: Callable[[Any], jnp.ndarray] | None = None,
    batch_len: int | None = None,
    return_index: bool = False,
    return_inverse: bool = False,
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

    Returns:
        Mask (bool array) or tuple (mask, index, inverse).
    """
    if key_fn is None:

        def key_fn(x):
            # Prefer aggregate-packed representation if available for density
            if hasattr(x, "packed"):
                return x.packed.words
            return x.uint32ed

    # 1. Generate keys for uniqueness
    try:
        unique_keys = jax.vmap(key_fn)(val)
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

    if key is not None and len(key) != batch_len:
        raise ValueError(f"key length {len(key)} must match batch_len {batch_len}")

    keys_flat = unique_keys.reshape(batch_len, -1)

    # 2. Wide Hashing to reduce sort columns (128-bit)
    hashes = _hash_to_wide(keys_flat)

    # 3. Handle filled: invalid items get MAX_KEY so they sort to the end
    if filled is not None:
        for i in range(len(hashes)):
            dtype = hashes[i].dtype
            max_val = jnp.array(jnp.iinfo(dtype).max, dtype=dtype)
            hashes[i] = jnp.where(filled, hashes[i], max_val)

    # 4. Prepare columns for lexsort
    # Lexsort expects [least_significant, ..., most_significant]
    sort_keys = []

    # Tertiary key: original index (stable tie-breaking if hashes and keys are equal)
    sort_keys.append(jnp.arange(batch_len, dtype=jnp.int32))

    if key is not None:
        # Secondary key: cost (minimize cost within same hash group)
        if filled is not None:
            inf_fill = jnp.array(jnp.inf, dtype=key.dtype)
            valid_key = jnp.where(filled, key, inf_fill)
        else:
            valid_key = key
        sort_keys.append(valid_key)

    # Primary key: hashes (to group identical states)
    # Reversed so the first hash is the most significant sorting factor.
    sort_keys.extend(reversed(hashes))

    # 5. Perform Lexsort
    perm = jnp.lexsort(sort_keys)

    # 6. Compute unique mask from sorted hashes
    sorted_hashes = [h[perm] for h in hashes]

    # Row i is unique if it differs from i-1 in ANY of the hashes
    diffs = [h[1:] != h[:-1] for h in sorted_hashes]
    is_diff = diffs[0]
    for d in diffs[1:]:
        is_diff = jnp.logical_or(is_diff, d)

    mask_sorted = jnp.concatenate([jnp.array([True]), is_diff])

    # 7. Filter out the "filled=False" group
    if filled is not None:
        is_valid_sorted = filled[perm]
        mask_sorted = jnp.logical_and(mask_sorted, is_valid_sorted)

    # 8. Map back to original order
    final_mask = jnp.zeros(batch_len, dtype=jnp.bool_).at[perm].set(mask_sorted)

    if not return_index and not return_inverse:
        return final_mask

    returns = (final_mask,)

    if return_index:
        unique_indices = perm[mask_sorted]
        returns += (unique_indices,)

    if return_inverse:
        group_id_sorted = jnp.cumsum(mask_sorted) - 1
        inv = jnp.zeros(batch_len, dtype=jnp.int32).at[perm].set(group_id_sorted)
        returns += (inv,)

    return returns
