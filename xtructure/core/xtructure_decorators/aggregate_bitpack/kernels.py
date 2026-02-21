"""Aggregate bitpack kernels.

This module contains a vectorized XLA implementation and a pallas-first
implementation for packing the aggregate bitstream.

The packing kernel computes each output word independently from a precomputed
contribution table, avoiding per-value scatter updates.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .spec import _AggWordContribTables


def _as_u32(x: Any) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.uint32)


def pack_words_all_xla(
    values_stream_u32: jax.Array, tables: _AggWordContribTables
) -> jax.Array:
    """Pack (flat_n, total_values) uint32 stream -> (flat_n, words_all_len) uint32 words."""
    values_stream_u32 = _as_u32(values_stream_u32)
    flat_n = int(values_stream_u32.shape[0])

    words_all_len = int(tables.value_idx.shape[0])
    max_contrib = int(tables.value_idx.shape[1]) if tables.value_idx.ndim == 2 else 0
    if words_all_len == 0:
        return jnp.zeros((flat_n, 0), dtype=jnp.uint32)
    if max_contrib == 0:
        return jnp.zeros((flat_n, words_all_len), dtype=jnp.uint32)

    total_values = int(tables.total_values)
    # Sentinel index points to a padded zero column.
    values_padded = jnp.concatenate(
        [values_stream_u32, jnp.zeros((flat_n, 1), dtype=jnp.uint32)], axis=1
    )

    value_idx = jnp.asarray(tables.value_idx, dtype=jnp.int32)
    safe_idx = jnp.where(value_idx >= 0, value_idx, jnp.int32(total_values))

    gathered = jnp.take(values_padded, safe_idx, axis=1)  # (flat_n, words, max_contrib)

    vb = jnp.asarray(tables.value_bit_start, dtype=jnp.uint32)[None, :, :]
    wb = jnp.asarray(tables.word_bit_start, dtype=jnp.uint32)[None, :, :]
    mask = jnp.asarray(tables.mask_u32, dtype=jnp.uint32)[None, :, :]

    parts = (jnp.right_shift(gathered, vb) & mask) << wb
    return jnp.bitwise_or.reduce(parts, axis=-1)


PallasBackend = Literal["triton", "mosaic_tpu"]


@lru_cache(maxsize=None)
def _get_pack_words_all_pallas(
    *,
    backend: PallasBackend,
    word_tile: int,
    words_all_len: int,
    max_contrib: int,
    total_values: int,
    value_idx: bytes,
    value_bit_start: bytes,
    word_bit_start: bytes,
    mask_u32: bytes,
) -> Any:
    """Build a pallas packing function.

    Note: We pass the numpy tables as bytes into the cache key to ensure
    correctness for different specs without storing large objects in the cache key.
    """

    if word_tile <= 0:
        raise ValueError("word_tile must be positive")
    if words_all_len < 0:
        raise ValueError("words_all_len must be non-negative")
    if max_contrib < 0:
        raise ValueError("max_contrib must be non-negative")

    from jax.experimental import pallas as pl

    # Rehydrate constant tables from bytes.
    # Keep these as NumPy arrays to avoid tracer leakage through the lru_cache.
    # We convert to JAX arrays inside the jitted wrapper.
    value_idx_np = (
        np.frombuffer(value_idx, dtype=np.int32)
        .reshape((words_all_len, max_contrib))
        .copy()
    )
    vb_np = (
        np.frombuffer(value_bit_start, dtype=np.uint8)
        .reshape((words_all_len, max_contrib))
        .copy()
    )
    wb_np = (
        np.frombuffer(word_bit_start, dtype=np.uint8)
        .reshape((words_all_len, max_contrib))
        .copy()
    )
    mask_np = (
        np.frombuffer(mask_u32, dtype=np.uint32)
        .reshape((words_all_len, max_contrib))
        .copy()
    )

    words_all_len_py = int(words_all_len)
    word_tile_py = int(word_tile)
    max_contrib_py = int(max_contrib)
    sentinel_py = int(total_values)

    def _kernel(values_ref, value_idx_ref, vb_ref, wb_ref, mask_ref, out_ref):
        pid_row = pl.program_id(axis=0)
        pid_block = pl.program_id(axis=1)

        w0 = pid_block * word_tile_py
        for off in range(word_tile_py):
            w = w0 + off
            do = w < words_all_len_py

            @pl.when(do)
            def _do_word():
                word = jnp.uint32(0)
                for t in range(max_contrib_py):
                    vi = jnp.asarray(value_idx_ref[w, t], dtype=jnp.int32)
                    safe_vi = jnp.where(vi >= 0, vi, jnp.int32(sentinel_py))
                    # In this JAX version, pallas uses ref indexing rather than pl.load/pl.store.
                    # safe_vi always points to an in-bounds column (we pad with a zero sentinel).
                    val = jnp.asarray(values_ref[pid_row, safe_vi], dtype=jnp.uint32)

                    vb = jnp.asarray(vb_ref[w, t], dtype=jnp.uint32)
                    wb = jnp.asarray(wb_ref[w, t], dtype=jnp.uint32)
                    m = jnp.asarray(mask_ref[w, t], dtype=jnp.uint32)
                    part = (jnp.right_shift(val, vb) & m) << wb
                    word = jnp.bitwise_or(word, part)

                out_ref[pid_row, w] = word

    @jax.jit
    def _pack(values_stream_u32: jax.Array) -> jax.Array:
        values_stream_u32 = jnp.asarray(values_stream_u32, dtype=jnp.uint32)
        flat_n = int(values_stream_u32.shape[0])

        if words_all_len == 0:
            return jnp.zeros((flat_n, 0), dtype=jnp.uint32)
        if max_contrib == 0:
            return jnp.zeros((flat_n, words_all_len), dtype=jnp.uint32)

        values_padded = jnp.concatenate(
            [values_stream_u32, jnp.zeros((flat_n, 1), dtype=jnp.uint32)], axis=1
        )

        value_idx_tbl = jnp.asarray(value_idx_np, dtype=jnp.int32)
        vb_tbl = jnp.asarray(vb_np, dtype=jnp.uint8)
        wb_tbl = jnp.asarray(wb_np, dtype=jnp.uint8)
        mask_tbl = jnp.asarray(mask_np, dtype=jnp.uint32)

        out_shape = jax.ShapeDtypeStruct((flat_n, words_all_len), jnp.uint32)
        grid_words = (words_all_len + word_tile - 1) // word_tile

        return pl.pallas_call(
            _kernel,
            grid=(flat_n, grid_words),
            out_shape=out_shape,
            backend=backend,
        )(
            values_padded,
            value_idx_tbl,
            vb_tbl,
            wb_tbl,
            mask_tbl,
        )

    return _pack


def pack_words_all_pallas(
    values_stream_u32: jax.Array,
    tables: _AggWordContribTables,
    *,
    backend: PallasBackend,
    word_tile: int,
) -> jax.Array:
    # Cache key: include table bytes so different specs don't collide.
    fn = _get_pack_words_all_pallas(
        backend=backend,
        word_tile=int(word_tile),
        words_all_len=int(tables.value_idx.shape[0]),
        max_contrib=int(tables.value_idx.shape[1]) if tables.value_idx.ndim == 2 else 0,
        total_values=int(tables.total_values),
        value_idx=tables.value_idx.tobytes(),
        value_bit_start=tables.value_bit_start.tobytes(),
        word_bit_start=tables.word_bit_start.tobytes(),
        mask_u32=tables.mask_u32.tobytes(),
    )
    return fn(values_stream_u32)
