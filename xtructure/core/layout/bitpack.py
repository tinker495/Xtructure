"""Bitpack Layout: single source of truth for packed-field sizing, layout
facts, low-level bit packing, and field-level pack/unpack.

This module owns:
- `default_unpack_dtype`: the **Packed Data Kind** policy (1/≤8/>8-bit).
- `packed_num_bytes`: byte-count math shared by **Type Layout** facts and IO.
- `to_uint8` / `from_uint8`: bit-level packing/unpacking primitives.
- `pack_field` / `unpack_field`: field-level pack/unpack against a
  **PackedFieldLayout**, used by both the in-memory bitpack accessor adapter
  and the IO save/load adapter — guaranteeing they cannot diverge.
- `build_aggregate_bitpack_layout`: aggregate eligibility builder.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np

from xtructure.core.dtype_facts import DTypeKind, dtype_kind

from .types import (
    AggregateBitpackLayout,
    AggregateBitpackReason,
    AggregateLeafLayout,
    AggregateViewFieldLayout,
    FieldLayout,
    PackedFieldLayout,
)


def default_unpack_dtype(bits: int) -> Any:
    if bits == 1:
        return jnp.bool_
    if bits <= 8:
        return jnp.uint8
    return jnp.uint32


def ceil_div(a: int, b: int) -> int:
    return int((a + b - 1) // b)


def packed_num_bytes(num_values: int, active_bits: int) -> int:
    """Return the number of uint8 bytes required to pack `num_values` values.

    Mirrors the block-aligned packing strategy of `to_uint8` for 3/5/6/7 etc.
    """
    if not isinstance(num_values, (int, np.integer)):
        raise TypeError(f"num_values must be an int, got {type(num_values).__name__}")
    if num_values < 0:
        raise ValueError(f"num_values must be non-negative, got {num_values}")
    if not isinstance(active_bits, int):
        raise TypeError(f"active_bits must be an int, got {type(active_bits).__name__}")
    if active_bits < 1 or active_bits > 32:
        raise ValueError(f"active_bits must be 1-32, got {active_bits}")

    if num_values == 0:
        return 0

    if active_bits == 8:
        return int(num_values)
    if active_bits == 1:
        return int((num_values + 7) // 8)
    if active_bits in (2, 4):
        values_per_byte = 8 // active_bits
        return int((num_values + values_per_byte - 1) // values_per_byte)

    block_bits = int(np.lcm(active_bits, 8))
    values_per_block = block_bits // active_bits
    bytes_per_block = block_bits // 8
    blocks = int((num_values + values_per_block - 1) // values_per_block)
    return int(blocks * bytes_per_block)


def compute_word_tail_layout(total_bits: int) -> tuple[int, int, int]:
    """Return (words_all_len, stored_words_len, tail_bytes)."""
    if total_bits < 0:
        raise ValueError("total_bits must be non-negative")
    words_all_len = ceil_div(total_bits, 32) if total_bits else 0
    rem_bits = total_bits % 32
    rem_bytes = ceil_div(rem_bits, 8) if rem_bits else 0
    if rem_bytes in (1, 2):
        tail_bytes = rem_bytes
        stored_words_len = max(words_all_len - 1, 0)
    else:
        tail_bytes = 0
        stored_words_len = words_all_len
    return words_all_len, stored_words_len, tail_bytes


def to_uint8(values: chex.Array, active_bits: int = 1) -> chex.Array:
    """Pack an array into a uint8 stream using `active_bits` per value.

    Args:
        values: Input array. For active_bits==1, can be bool or integer (0/!=0).
            For active_bits>1, must be integer.
        active_bits: Bits per value in [1, 32].

    Returns:
        A 1D uint8 array of packed bytes.
    """
    assert 1 <= active_bits <= 32, f"active_bits must be 1-32, got {active_bits}"
    value_kind = dtype_kind(values.dtype)

    if active_bits == 1:
        flatten_input = values.reshape((-1,))
        if value_kind not in (DTypeKind.BOOL, DTypeKind.UINT, DTypeKind.INT):
            raise TypeError(
                f"values must be bool or integer array for active_bits=1, got DType Kind {value_kind.value!r}"
            )
        if value_kind is not DTypeKind.BOOL:
            flatten_input = flatten_input != 0
        return jnp.packbits(flatten_input, axis=-1, bitorder="little")

    if value_kind not in (DTypeKind.UINT, DTypeKind.INT):
        raise TypeError(
            f"values must be integer array for active_bits={active_bits}, got DType Kind {value_kind.value!r}"
        )

    values_flat = values.reshape((-1,))

    if active_bits == 8:
        return values_flat.astype(jnp.uint8)

    if active_bits in (2, 4):
        values_per_byte = 8 // active_bits
        padding = (values_per_byte - (values_flat.size % values_per_byte)) % values_per_byte
        if padding:
            values_flat = jnp.concatenate(
                [values_flat, jnp.zeros((padding,), dtype=values_flat.dtype)]
            )
        grouped = values_flat.reshape((-1, values_per_byte))

        def pack_group(group):
            out = jnp.uint8(0)
            for i in range(values_per_byte):
                out = out | (group[i].astype(jnp.uint8) << jnp.uint8(i * active_bits))
            return out

        return jax.vmap(pack_group)(grouped)

    L = int(np.lcm(active_bits, 8))
    num_values_per_block = L // active_bits
    num_bytes_per_block = L // 8

    padding = (
        num_values_per_block - (values_flat.size % num_values_per_block)
    ) % num_values_per_block
    if padding:
        values_flat = jnp.concatenate([values_flat, jnp.zeros((padding,), dtype=values_flat.dtype)])
    grouped = values_flat.reshape((-1, num_values_per_block))

    if L <= 32:

        def pack_block(group):
            acc = jnp.uint32(0)
            for i in range(num_values_per_block):
                acc = acc | (group[i].astype(jnp.uint32) << jnp.uint32(i * active_bits))
            return jnp.array(
                [(acc >> jnp.uint32(8 * j)) & jnp.uint32(0xFF) for j in range(num_bytes_per_block)],
                dtype=jnp.uint8,
            )

        packed = jax.vmap(pack_block)(grouped)
        return packed.reshape((-1,))

    def pack_block(group):
        packed_bytes = jnp.zeros((num_bytes_per_block,), dtype=jnp.uint8)
        acc = jnp.uint32(0)
        bits_in_acc = 0
        byte_idx = 0
        for i in range(num_values_per_block):
            acc = acc | (group[i].astype(jnp.uint32) << jnp.uint32(bits_in_acc))
            bits_in_acc += active_bits
            while bits_in_acc >= 8:
                packed_bytes = packed_bytes.at[byte_idx].set(
                    (acc & jnp.uint32(0xFF)).astype(jnp.uint8)
                )
                acc = acc >> jnp.uint32(8)
                bits_in_acc -= 8
                byte_idx += 1
        if byte_idx < num_bytes_per_block:
            packed_bytes = packed_bytes.at[byte_idx].set((acc & jnp.uint32(0xFF)).astype(jnp.uint8))
        return packed_bytes

    packed = jax.vmap(pack_block)(grouped)
    return packed.reshape((-1,))


def from_uint8(
    packed_bytes: chex.Array, target_shape: tuple[int, ...], active_bits: int = 1
) -> chex.Array:
    """Unpack a uint8 stream back into an array of shape `target_shape`.

    For active_bits==1, returns bool. For active_bits>1, returns
    `default_unpack_dtype(active_bits)` values; caller may cast.
    """
    packed_bytes = jnp.asarray(packed_bytes, dtype=jnp.uint8).reshape((-1,))
    assert 1 <= active_bits <= 32, f"active_bits must be 1-32, got {active_bits}"

    num_target_elements = int(np.prod(target_shape))
    assert num_target_elements >= 0, "target_shape must have non-negative product"
    if num_target_elements == 0:
        return jnp.zeros(target_shape, dtype=default_unpack_dtype(active_bits))

    if active_bits == 1:
        bits = jnp.unpackbits(packed_bytes, count=num_target_elements, bitorder="little")
        return bits.reshape(target_shape).astype(jnp.bool_)

    if active_bits == 8:
        return packed_bytes[:num_target_elements].reshape(target_shape)

    if active_bits in (2, 4):
        values_per_byte = 8 // active_bits
        mask = jnp.uint8((1 << active_bits) - 1)

        def unpack_byte(b):
            vals = []
            for i in range(values_per_byte):
                vals.append((b >> jnp.uint8(i * active_bits)) & mask)
            return jnp.array(vals, dtype=jnp.uint8)

        groups = jax.vmap(unpack_byte)(packed_bytes)
        all_values = groups.reshape((-1,))
        return all_values[:num_target_elements].reshape(target_shape)

    L = int(np.lcm(active_bits, 8))
    num_values_per_block = L // active_bits
    num_bytes_per_block = L // 8
    mask = jnp.uint32(0xFFFFFFFF) if active_bits == 32 else jnp.uint32((1 << active_bits) - 1)

    total_blocks = (packed_bytes.size + num_bytes_per_block - 1) // num_bytes_per_block
    total_bytes = total_blocks * num_bytes_per_block
    if total_bytes != packed_bytes.size:
        packed_bytes = jnp.pad(packed_bytes, (0, total_bytes - packed_bytes.size), mode="constant")
    grouped = packed_bytes.reshape((-1, num_bytes_per_block))

    if L <= 32:

        def unpack_block(byte_group):
            acc = jnp.uint32(0)
            for j in range(num_bytes_per_block):
                acc = acc | (byte_group[j].astype(jnp.uint32) << jnp.uint32(8 * j))
            vals = [
                (acc >> jnp.uint32(i * active_bits)) & mask for i in range(num_values_per_block)
            ]
            return jnp.array(vals, dtype=default_unpack_dtype(active_bits))

        blocks = jax.vmap(unpack_block)(grouped)
        all_values = blocks.reshape((-1,))
        return all_values[:num_target_elements].reshape(target_shape)

    def unpack_block(byte_group):
        out_dtype = default_unpack_dtype(active_bits)
        vals = jnp.zeros((num_values_per_block,), dtype=out_dtype)
        acc = jnp.uint32(0)
        bits_in_acc = 0
        byte_idx = 0
        for i in range(num_values_per_block):
            while bits_in_acc < active_bits and byte_idx < num_bytes_per_block:
                acc = acc | (byte_group[byte_idx].astype(jnp.uint32) << jnp.uint32(bits_in_acc))
                bits_in_acc += 8
                byte_idx += 1
            vals = vals.at[i].set((acc & mask).astype(out_dtype))
            acc = acc >> jnp.uint32(active_bits)
            bits_in_acc -= active_bits
        return vals

    blocks = jax.vmap(unpack_block)(grouped)
    all_values = blocks.reshape((-1,))
    return all_values[:num_target_elements].reshape(target_shape)


def pack_field(
    unpacked_value: Any,
    packed_layout: PackedFieldLayout,
    batch_shape: tuple[int, ...],
):
    """Pack one field's unpacked array into its **PackedFieldLayout** byte stream.

    Used by both the in-memory bitpack accessor adapter and the IO save adapter
    so the two paths cannot diverge.
    """
    packed_bits = packed_layout.packed_bits
    unpacked_shape = packed_layout.unpacked_intrinsic_shape

    arr = jnp.asarray(unpacked_value)
    if arr.shape[: len(batch_shape)] != batch_shape:
        raise ValueError(
            f"Unpacked value batch_shape mismatch: expected {batch_shape}, "
            f"got {arr.shape[:len(batch_shape)]}."
        )
    if arr.shape[len(batch_shape) :] != unpacked_shape:
        raise ValueError(
            f"Unpacked value trailing shape mismatch: expected {unpacked_shape}, "
            f"got {arr.shape[len(batch_shape):]}."
        )

    expected_packed_len = packed_layout.packed_byte_count
    flat = arr.reshape((-1,) + unpacked_shape)

    def _pack_row(row):
        return to_uint8(row, active_bits=packed_bits)

    packed_flat = jax.vmap(_pack_row)(flat)
    return packed_flat.reshape(batch_shape + (expected_packed_len,)).astype(jnp.uint8)


def unpack_field(
    packed_value: Any,
    packed_layout: PackedFieldLayout,
    batch_shape: tuple[int, ...],
):
    """Unpack one field's byte stream back into its logical unpacked array."""
    packed_bits = packed_layout.packed_bits
    unpacked_shape = packed_layout.unpacked_intrinsic_shape
    unpacked_dtype = packed_layout.unpacked_dtype
    packed_arr = jnp.asarray(packed_value, dtype=jnp.uint8)
    expected_packed_len = packed_layout.packed_byte_count

    if packed_arr.shape[-1] != expected_packed_len:
        raise ValueError(
            f"Packed leaf has wrong trailing length: expected {expected_packed_len}, "
            f"got {packed_arr.shape[-1]}."
        )

    flat = packed_arr.reshape((-1, expected_packed_len))

    def _unpack_row(row):
        out = from_uint8(row, target_shape=unpacked_shape, active_bits=packed_bits)
        if packed_bits > 1:
            try:
                out = out.astype(unpacked_dtype)
            except TypeError:
                pass
        return out

    unpacked_flat = jax.vmap(_unpack_row)(flat)
    return unpacked_flat.reshape(batch_shape + unpacked_shape)


def build_aggregate_bitpack_layout(
    root_cls: type, fields: tuple[FieldLayout, ...]
) -> AggregateBitpackLayout:
    """Compute current aggregate bitpack eligibility without changing constraints."""
    leaves: list[AggregateLeafLayout] = []
    view_fields_by_owner: dict[type, list[AggregateViewFieldLayout]] = {}
    bit_offset = 0

    def fail(reason: str, reason_kind: AggregateBitpackReason) -> AggregateBitpackLayout:
        return AggregateBitpackLayout(
            eligible=False,
            reason=reason,
            reason_kind=reason_kind,
            leaves=(),
            total_bits=0,
            words_all_len=0,
            stored_words_len=0,
            tail_bytes=0,
            view_fields_by_owner=(),
        )

    def walk(
        owner_type: type,
        cls_fields: tuple[FieldLayout, ...],
        prefix: tuple[str, ...] = (),
    ) -> tuple[str, AggregateBitpackReason] | None:
        nonlocal bit_offset
        owner_view_fields = view_fields_by_owner.setdefault(owner_type, [])
        for field in cls_fields:
            path = prefix + (field.name,)
            if field.is_nested:
                if field.intrinsic_shape != ():
                    return (
                        "aggregate_bitpack currently supports only scalar nested fields. "
                        f"Got intrinsic_shape={field.intrinsic_shape} on {'.'.join(path)}.",
                        AggregateBitpackReason.SCALAR_NESTED,
                    )
                owner_view_fields.append(
                    AggregateViewFieldLayout(
                        owner_type=owner_type,
                        name=field.name,
                        path=path,
                        is_nested=True,
                        nested_type=field.nested_type,
                        unpack_dtype=None,
                        unpacked_shape=(),
                    )
                )
                from .type_layout import get_type_layout

                nested_layout = get_type_layout(field.nested_type)  # type: ignore[arg-type]
                nested_reason = walk(nested_layout.cls, nested_layout.fields, path)
                if nested_reason is not None:
                    return nested_reason
                continue

            if field.bits is None:
                return (
                    "aggregate_bitpack requires FieldDescriptor.bits on every primitive leaf. "
                    f"Missing on {'.'.join(path)}.",
                    AggregateBitpackReason.MISSING_BITS,
                )

            bits = int(field.bits)
            unpacked_shape = tuple(field.intrinsic_shape)
            nvalues = (
                int(np.prod(np.array(unpacked_shape, dtype=np.int64))) if unpacked_shape else 1
            )
            bit_len = int(nvalues * bits)
            unpack_dtype = default_unpack_dtype(bits)
            leaves.append(
                AggregateLeafLayout(
                    path=path,
                    bits=bits,
                    unpacked_shape=unpacked_shape,
                    nvalues=nvalues,
                    bit_offset=bit_offset,
                    bit_len=bit_len,
                    unpack_dtype=unpack_dtype,
                    declared_dtype=field.dtype,
                )
            )
            owner_view_fields.append(
                AggregateViewFieldLayout(
                    owner_type=owner_type,
                    name=field.name,
                    path=path,
                    is_nested=False,
                    nested_type=None,
                    unpack_dtype=unpack_dtype,
                    unpacked_shape=unpacked_shape,
                )
            )
            bit_offset += bit_len
        return None

    walk_result = walk(root_cls, fields)
    if walk_result is not None:
        return fail(*walk_result)
    if not leaves:
        return fail(
            "aggregate_bitpack requires at least one primitive leaf.",
            AggregateBitpackReason.NO_LEAVES,
        )

    total_bits = int(bit_offset)
    words_all_len, stored_words_len, tail_bytes = compute_word_tail_layout(total_bits)
    return AggregateBitpackLayout(
        eligible=True,
        reason=None,
        reason_kind=None,
        leaves=tuple(leaves),
        total_bits=total_bits,
        words_all_len=words_all_len,
        stored_words_len=stored_words_len,
        tail_bytes=tail_bytes,
        view_fields_by_owner=tuple(
            (owner, tuple(owner_fields)) for owner, owner_fields in view_fields_by_owner.items()
        ),
    )
