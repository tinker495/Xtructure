"""Optimized unique_mask implementation using wide hashing and Lexsort."""

from __future__ import annotations

from typing import Any, Callable, Union

import jax
import jax.numpy as jnp

from ....dtype_facts import DTypeKind, dtype_kind
from ....xtructure_decorators import Xtructurable


def _pack_uint8_rows(rows: jnp.ndarray) -> jnp.ndarray:
    """Pack ``(batch, bytes)`` uint8 rows into per-row uint32 lanes."""
    rows = jnp.asarray(rows, dtype=jnp.uint8)
    batch_len, row_len = rows.shape
    pad_len = (-row_len) % 4
    if pad_len:
        rows = jnp.pad(rows, ((0, 0), (0, pad_len)), mode="constant", constant_values=0)
    if rows.shape[1] == 0:
        return jnp.zeros((batch_len, 0), dtype=jnp.uint32)
    chunks = rows.reshape(batch_len, -1, 4)
    return jax.lax.bitcast_convert_type(chunks, jnp.uint32).reshape(batch_len, -1)


def _pack_uint16_rows(rows: jnp.ndarray) -> jnp.ndarray:
    """Pack ``(batch, lanes)`` uint16 rows into per-row uint32 lanes."""
    rows = jnp.asarray(rows, dtype=jnp.uint16)
    batch_len, row_len = rows.shape
    pad_len = (-row_len) % 2
    if pad_len:
        rows = jnp.pad(rows, ((0, 0), (0, pad_len)), mode="constant", constant_values=0)
    if rows.shape[1] == 0:
        return jnp.zeros((batch_len, 0), dtype=jnp.uint32)
    pairs = rows.reshape(batch_len, -1, 2)
    lo = pairs[:, :, 0].astype(jnp.uint32)
    hi = pairs[:, :, 1].astype(jnp.uint32)
    return lo | (hi << jnp.uint32(16))


def _split_uint64_rows(rows: jnp.ndarray) -> jnp.ndarray:
    """Split ``(batch, lanes)`` uint64 rows into interleaved uint32 lanes."""
    rows = jnp.asarray(rows, dtype=jnp.uint64)
    lo = (rows & jnp.uint64(0xFFFFFFFF)).astype(jnp.uint32)
    hi = (rows >> jnp.uint64(32)).astype(jnp.uint32)
    return jnp.stack((lo, hi), axis=-1).reshape(rows.shape[0], -1)


def _leaf_to_uint32_rows(leaf: Any, batch_len: int) -> jnp.ndarray:
    """Convert a batched PyTree leaf into row-wise uint32 lanes.

    This mirrors ``jax.vmap(lambda x: x.uint32ed)(val)`` for the default
    unique key without rematerializing one scalar PyTree conversion per row.
    Padding is applied per row and per leaf, matching the scalar ``uint32ed``
    representation used by the hash decorator.
    """
    leaf = jnp.asarray(leaf)
    if leaf.ndim == 0 or leaf.shape[0] != batch_len:
        raise ValueError(
            "default unique key generation expects every dynamic leaf to carry "
            f"leading batch dimension {batch_len}; got leaf shape {leaf.shape}."
        )

    rows = leaf.reshape(batch_len, -1)
    kind = dtype_kind(leaf.dtype)

    if kind is DTypeKind.BOOL:
        return _pack_uint8_rows(rows.astype(jnp.uint8))

    if kind in (DTypeKind.UINT, DTypeKind.INT):
        bits = jnp.iinfo(leaf.dtype).bits
        if bits == 8:
            return _pack_uint8_rows(rows.astype(jnp.uint8))
        if bits == 16:
            return _pack_uint16_rows(rows.astype(jnp.uint16))
        if bits == 32:
            return rows.astype(jnp.uint32)
        if bits == 64:
            return _split_uint64_rows(rows.astype(jnp.uint64))

    if kind is DTypeKind.FLOAT:
        if leaf.dtype == jnp.float32:
            return jax.lax.bitcast_convert_type(rows, jnp.uint32).reshape(batch_len, -1)
        if leaf.dtype == jnp.float64:
            uint64_rows = jax.lax.bitcast_convert_type(rows, jnp.uint64).reshape(batch_len, -1)
            return _split_uint64_rows(uint64_rows)
        if leaf.dtype in (jnp.float16, jnp.bfloat16):
            uint16_rows = jax.lax.bitcast_convert_type(rows, jnp.uint16).reshape(batch_len, -1)
            return _pack_uint16_rows(uint16_rows)

    raise TypeError(f"Unsupported DType Kind for unique key encoding: {leaf.dtype!r}.")


def _batched_uint32_keys(val: Xtructurable, batch_len: int) -> jnp.ndarray:
    """Return row-wise default ``uint32ed`` keys for a batched Xtructurable."""
    uint32_leaves = [
        _leaf_to_uint32_rows(leaf, batch_len) for leaf in jax.tree_util.tree_leaves(val)
    ]
    if not uint32_leaves:
        return jnp.zeros((batch_len, 0), dtype=jnp.uint32)
    return jnp.concatenate(uint32_leaves, axis=1)


def _hash_to_wide(keys: jnp.ndarray) -> list[jnp.ndarray]:
    """Hash (N, K) uint32 keys into a fixed-width wide hash."""
    n, k = keys.shape

    if k == 0:
        return [jnp.zeros((n,), dtype=jnp.uint32)]

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


def _compact_unique_indices(
    perm: jnp.ndarray,
    mask_sorted: jnp.ndarray,
    batch_len: int,
) -> jnp.ndarray:
    """Return fixed-size selected indices, padded with ``batch_len``.

    Boolean indexing would create a dynamic-length result and fail under JIT.
    This static compaction keeps JAX shapes fixed while preserving the selected
    index order from the sorted groups.
    """
    sentinel = jnp.asarray(batch_len, dtype=perm.dtype)
    slots = jnp.cumsum(mask_sorted.astype(jnp.int32)) - 1
    slots = jnp.maximum(slots, 0)
    values = jnp.where(mask_sorted, perm, sentinel)
    return jnp.full((batch_len,), sentinel, dtype=perm.dtype).at[slots].min(values)


def _unique_indices_from_mask(final_mask: jnp.ndarray, batch_len: int) -> jnp.ndarray:
    indices = jnp.arange(batch_len, dtype=jnp.int32)
    return _compact_unique_indices(indices, final_mask, batch_len)


def _unique_mask_exact(
    unique_keys: jnp.ndarray,
    key: jnp.ndarray | None,
    filled: jnp.ndarray | None,
    batch_len: int,
    return_index: bool,
    return_inverse: bool,
) -> Union[jnp.ndarray, tuple]:
    """Exact small-key path backed by ``jnp.unique``.

    This avoids wide-hash overhead for narrow keys and preserves full precision
    for custom key functions that return non-uint32 values.
    """
    unique_input = unique_keys
    if filled is not None:
        flat_keys = unique_keys.reshape(batch_len, -1)
        filled_col = filled.reshape(batch_len, 1).astype(flat_keys.dtype)
        unique_input = jnp.concatenate([flat_keys, filled_col], axis=1)

    if key is None and not return_inverse:
        _, unique_indices = jnp.unique(
            unique_input,
            axis=0,
            size=batch_len,
            return_index=True,
        )
        final_mask = jnp.zeros(batch_len, dtype=jnp.bool_).at[unique_indices].set(True)
        if filled is not None:
            final_mask = jnp.logical_and(final_mask, filled)

        if not return_index:
            return final_mask
        return (final_mask, _unique_indices_from_mask(final_mask, batch_len))

    _, inv = jnp.unique(
        unique_input,
        axis=0,
        size=batch_len,
        return_inverse=True,
    )

    batch_idx = jnp.arange(batch_len, dtype=jnp.int32)

    if key is None:
        candidate_indices = (
            jnp.where(filled, batch_idx, batch_len) if filled is not None else batch_idx
        )
    else:
        if filled is not None:
            inf_fill = jnp.array(jnp.inf, dtype=key.dtype)
            masked_key = jnp.where(filled, key, inf_fill)
        else:
            masked_key = key

        min_keys = jnp.full((batch_len,), jnp.inf, dtype=masked_key.dtype).at[inv].min(masked_key)
        is_min_key = masked_key == min_keys[inv]
        candidate_mask = jnp.logical_and(is_min_key, filled) if filled is not None else is_min_key
        candidate_indices = jnp.where(candidate_mask, batch_idx, batch_len)

    representative_per_group = (
        jnp.full((batch_len,), batch_len, dtype=jnp.int32).at[inv].min(candidate_indices)
    )
    representative_per_group = jnp.where(
        representative_per_group == batch_len, 0, representative_per_group
    )
    representative_for_item = representative_per_group[inv]
    final_mask = batch_idx == representative_for_item
    if filled is not None:
        final_mask = jnp.logical_and(final_mask, filled)
    elif key is not None:
        final_mask = jnp.logical_and(final_mask, key < jnp.inf)

    if not return_index and not return_inverse:
        return final_mask

    returns = (final_mask,)
    if return_index:
        returns += (_unique_indices_from_mask(final_mask, batch_len),)
    if return_inverse:
        returns += (inv,)
    return returns


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

    Adaptive implementation using exact ``jnp.unique`` for narrow or non-uint32
    keys, and wide hashing + lexsort for wider uint32 keys. The wide path
    reduces multi-column keys into a fixed-width representation (128-bit),
    minimizing sorting passes and comparison overhead while maintaining
    near-zero collision probability for the default uint32 encoding.

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
    if batch_len is None:
        try:
            batch_len = val.shape.batch[0]
        except Exception as e:
            raise ValueError(f"key_fn failed to generate hashable keys: {e}")

    if key is not None and len(key) != batch_len:
        raise ValueError(f"key length {len(key)} must match batch_len {batch_len}")

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

    # 1. Generate keys for uniqueness.
    #
    # The default key is exactly each row's ``uint32ed`` representation.  Build
    # it directly from batched leaves so the common path avoids a per-row vmap
    # around the scalar hash adapter.  Custom key functions keep the previous
    # vmap contract.
    try:
        if key_fn is None:
            if not hasattr(val, "uint32ed"):
                raise AttributeError("value does not expose uint32ed")
            unique_keys = _batched_uint32_keys(val, batch_len)
        else:
            unique_keys = jax.vmap(key_fn)(val)
    except Exception as e:
        raise ValueError(f"key_fn failed to generate hashable keys: {e}")

    keys_flat = unique_keys.reshape(batch_len, -1)

    if keys_flat.shape[1] > 0 and (keys_flat.dtype != jnp.uint32 or keys_flat.shape[1] <= 2):
        return _unique_mask_exact(
            unique_keys=unique_keys,
            key=key,
            filled=filled,
            batch_len=batch_len,
            return_index=return_index,
            return_inverse=return_inverse,
        )

    # 2. Wide Hashing to reduce sort columns (128-bit)
    hashes = _hash_to_wide(keys_flat)

    # 3. Handle filled: invalid items get MAX_KEY so they sort to the end
    if filled is not None:
        for i in range(len(hashes)):
            dtype = hashes[i].dtype
            max_val = jnp.array(jnp.iinfo(dtype).max, dtype=dtype)
            hashes[i] = jnp.where(filled, hashes[i], max_val)

    # 4. Prepare columns for lexsort
    sort_keys = []

    if key is not None:
        if filled is not None:
            inf_fill = jnp.array(jnp.inf, dtype=key.dtype)
            valid_key = jnp.where(filled, key, inf_fill)
        else:
            valid_key = key
        sort_keys.append(valid_key)
    else:
        # Stable sort: prefer earlier index
        sort_keys.append(jnp.arange(batch_len, dtype=jnp.int32))

    # Add hashes as primary (last in lexsort list).
    # Reversed so the first hash is most significant.
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
        unique_indices = _compact_unique_indices(perm, mask_sorted, batch_len)
        returns += (unique_indices,)

    if return_inverse:
        group_id_sorted = jnp.cumsum(mask_sorted) - 1
        inv = jnp.zeros(batch_len, dtype=jnp.int32).at[perm].set(group_id_sorted)
        returns += (inv,)

    return returns
