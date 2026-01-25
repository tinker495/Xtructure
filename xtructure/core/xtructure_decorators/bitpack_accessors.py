"""Decorator that adds in-memory bitpack accessors for packed fields.

Packed fields are those whose FieldDescriptor declares `packed_bits` plus
`unpacked_dtype` and `unpacked_intrinsic_shape`. The stored leaf is expected
to be a uint8 byte-stream (typically created via FieldDescriptor.packed_tensor),
and the decorator adds:

- `<field>_unpacked` property: returns logical array of shape batch + unpacked_shape
- `set_unpacked(**kwargs)` method: packs provided logical arrays and returns a new instance
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Type, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from xtructure.core.field_descriptors import FieldDescriptor, get_field_descriptors
from xtructure.core.xtructure_decorators.aggregate_bitpack.bitpack import (
    packed_num_bytes,
)
from xtructure.core.xtructure_decorators.aggregate_bitpack.kernels import (
    pack_words_all_xla,
)
from xtructure.core.xtructure_decorators.aggregate_bitpack.spec import (
    _AggLeafSpec,
    _build_word_contrib_tables,
)

T = TypeVar("T")


def _ceil_div(a: int, b: int) -> int:
    return int((a + b - 1) // b)


@lru_cache(maxsize=None)
def _get_bitpack_pack_tables(*, packed_bits: int, num_values: int):
    """Precompute word contribution tables for packed_tensor packing."""
    if packed_bits < 1 or packed_bits > 32:
        raise ValueError(f"packed_bits must be 1..32, got {packed_bits}")
    if num_values < 0:
        raise ValueError(f"num_values must be non-negative, got {num_values}")

    expected_packed_len = packed_num_bytes(num_values, packed_bits)
    if num_values == 0:
        return None, expected_packed_len, 0, 0
    if packed_bits == 1:
        return None, expected_packed_len, num_values, 0

    L = int(np.lcm(packed_bits, 8))
    values_per_block = int(L // packed_bits)
    padded_values = _ceil_div(num_values, values_per_block) * values_per_block
    total_bits = int(padded_values * packed_bits)
    words_all_len = int(_ceil_div(total_bits, 32)) if total_bits else 0

    spec = _AggLeafSpec(
        path=("leaf",),
        bits=int(packed_bits),
        unpacked_shape=(padded_values,),
        nvalues=int(padded_values),
        bit_offset=0,
        bit_len=int(padded_values * packed_bits),
        unpack_dtype=jnp.uint32,
        declared_dtype=jnp.uint32,
    )
    tables = _build_word_contrib_tables([spec], words_all_len=words_all_len)
    return tables, expected_packed_len, padded_values, words_all_len


def _is_packed_descriptor(descriptor: FieldDescriptor) -> bool:
    return (
        descriptor.packed_bits is not None
        and descriptor.unpacked_intrinsic_shape is not None
        and descriptor.unpacked_dtype is not None
    )


def _unpack_value(packed_value: Any, descriptor: FieldDescriptor, batch_shape: tuple[int, ...]):
    packed_bits = int(descriptor.packed_bits)  # type: ignore[arg-type]
    unpacked_shape = tuple(descriptor.unpacked_intrinsic_shape)  # type: ignore[arg-type]
    unpacked_dtype = descriptor.unpacked_dtype
    if unpacked_dtype is None:
        if packed_bits == 1:
            unpacked_dtype = jnp.bool_
        elif packed_bits <= 8:
            unpacked_dtype = jnp.uint8
        else:
            unpacked_dtype = jnp.uint32

    packed_arr = jnp.asarray(packed_value, dtype=jnp.uint8)
    num_values = int(np.prod(np.array(unpacked_shape, dtype=np.int64)))
    expected_packed_len = packed_num_bytes(num_values, packed_bits)

    if packed_arr.shape[-1] != expected_packed_len:
        raise ValueError(
            f"Packed leaf has wrong trailing length: expected {expected_packed_len}, "
            f"got {packed_arr.shape[-1]}."
        )

    flat = packed_arr.reshape((-1, expected_packed_len))

    if num_values == 0:
        out_dtype = jnp.bool_ if packed_bits == 1 else jnp.uint8
        out = jnp.zeros((flat.shape[0], 0), dtype=out_dtype)
        return out.reshape(batch_shape + unpacked_shape)

    if packed_bits == 1:
        bits = jnp.unpackbits(flat, count=num_values, bitorder="little", axis=1)
        out = bits.astype(jnp.bool_)
        if unpacked_dtype is not None and unpacked_dtype != jnp.bool_:
            out = out.astype(unpacked_dtype)
        return out.reshape(batch_shape + unpacked_shape)

    # Convert packed bytes -> uint32 words (per row)
    pad_bytes = (-expected_packed_len) % 4
    if pad_bytes:
        flat = jnp.pad(flat, ((0, 0), (0, pad_bytes)), mode="constant", constant_values=0)

    words = jax.lax.bitcast_convert_type(flat.reshape((flat.shape[0], -1, 4)), jnp.uint32).reshape(
        (flat.shape[0], -1)
    )
    words_padded = jnp.concatenate(
        [words, jnp.zeros((words.shape[0], 1), dtype=jnp.uint32)], axis=1
    )

    bit_pos = (jnp.arange(num_values, dtype=jnp.uint32) * jnp.uint32(packed_bits)).astype(
        jnp.uint32
    )
    word_idx = jnp.right_shift(bit_pos, jnp.uint32(5)).astype(jnp.int32)
    shift = (bit_pos & jnp.uint32(31)).astype(jnp.uint32)

    idx2d = jnp.broadcast_to(word_idx[None, :], (words.shape[0], num_values))
    w0 = jnp.take_along_axis(words_padded, idx2d, axis=1)
    w1 = jnp.take_along_axis(words_padded, idx2d + 1, axis=1)

    shift2d = shift[None, :]
    low = jnp.right_shift(w0, shift2d)
    high = jnp.where(shift2d == 0, jnp.uint32(0), jnp.left_shift(w1, jnp.uint32(32) - shift2d))

    mask = jnp.uint32(0xFFFFFFFF) if packed_bits == 32 else jnp.uint32((1 << packed_bits) - 1)
    vals = jnp.bitwise_and(jnp.bitwise_or(low, high), mask)

    out_dtype = jnp.uint8 if packed_bits <= 8 else jnp.uint32
    out = vals.astype(out_dtype)
    if unpacked_dtype is not None and unpacked_dtype != out_dtype:
        try:
            out = out.astype(unpacked_dtype)
        except TypeError:
            pass
    return out.reshape(batch_shape + unpacked_shape)


def _pack_value(unpacked_value: Any, descriptor: FieldDescriptor, batch_shape: tuple[int, ...]):
    packed_bits = int(descriptor.packed_bits)  # type: ignore[arg-type]
    unpacked_shape = tuple(descriptor.unpacked_intrinsic_shape)  # type: ignore[arg-type]

    arr = jnp.asarray(unpacked_value)
    if arr.shape[: len(batch_shape)] != batch_shape:
        raise ValueError(
            f"Unpacked value batch_shape mismatch: expected {batch_shape}, got {arr.shape[:len(batch_shape)]}."
        )
    if arr.shape[len(batch_shape) :] != unpacked_shape:
        raise ValueError(
            f"Unpacked value trailing shape mismatch: expected {unpacked_shape}, got {arr.shape[len(batch_shape):]}."
        )

    num_values = int(np.prod(np.array(unpacked_shape, dtype=np.int64)))
    expected_packed_len = packed_num_bytes(num_values, packed_bits)

    flat = arr.reshape((-1, num_values))

    if num_values == 0:
        return jnp.zeros((flat.shape[0], expected_packed_len), dtype=jnp.uint8).reshape(
            batch_shape + (expected_packed_len,)
        )

    if packed_bits == 1:
        bits = flat
        if bits.dtype != jnp.bool_:
            bits = bits != 0
        packed = jnp.packbits(bits, axis=1, bitorder="little")
        return packed.reshape(batch_shape + (expected_packed_len,)).astype(jnp.uint8)

    # General pack: pad values to lcm(active_bits, 8) block size and pack bitstream.
    tables, expected_len_cached, padded_values, words_all_len = _get_bitpack_pack_tables(
        packed_bits=int(packed_bits),
        num_values=int(num_values),
    )
    if expected_len_cached != expected_packed_len:
        raise ValueError("packed_num_bytes mismatch for packed_tensor")

    if padded_values != num_values:
        flat = jnp.pad(
            flat, ((0, 0), (0, int(padded_values - num_values))), mode="constant", constant_values=0
        )

    if words_all_len == 0 or tables is None:
        return jnp.zeros((flat.shape[0], expected_packed_len), dtype=jnp.uint8).reshape(
            batch_shape + (expected_packed_len,)
        )

    words = pack_words_all_xla(jnp.asarray(flat, dtype=jnp.uint32), tables)
    bytes_all = jax.lax.bitcast_convert_type(words, jnp.uint8).reshape((flat.shape[0], -1))
    packed = bytes_all[:, :expected_packed_len]
    return packed.reshape(batch_shape + (expected_packed_len,)).astype(jnp.uint8)


def add_bitpack_accessors(cls: Type[T]) -> Type[T]:
    """Attach unpacked accessors for any packed fields defined on the class."""
    field_descriptors: dict[str, FieldDescriptor] = get_field_descriptors(cls)

    packed_fields = [name for name, fd in field_descriptors.items() if _is_packed_descriptor(fd)]
    if not packed_fields:
        return cls

    for field_name in packed_fields:
        fd = field_descriptors[field_name]

        def _make_unpacked_property(name: str, descriptor: FieldDescriptor):
            def _prop(self):
                batch = getattr(self.shape, "batch", ())
                if batch == -1:
                    raise TypeError(
                        f"{cls.__name__} is UNSTRUCTURED (shape.batch == -1); "
                        f"cannot infer batch for unpacking '{name}'."
                    )
                return _unpack_value(getattr(self, name), descriptor, batch)

            return property(_prop)

        setattr(cls, f"{field_name}_unpacked", _make_unpacked_property(field_name, fd))

    def set_unpacked(self, **kwargs):
        """Return a new instance with provided packed fields updated from unpacked arrays."""
        batch = getattr(self.shape, "batch", ())
        if batch == -1:
            raise TypeError(
                f"{cls.__name__} is UNSTRUCTURED (shape.batch == -1); cannot set_unpacked."
            )

        updates = {}
        for name, value in kwargs.items():
            if name not in field_descriptors:
                raise KeyError(f"Unknown field '{name}' for {cls.__name__}.")
            descriptor = field_descriptors[name]
            if not _is_packed_descriptor(descriptor):
                raise TypeError(
                    f"Field '{name}' is not a packed field (missing packed_bits/unpacked metadata)."
                )
            updates[name] = _pack_value(value, descriptor, batch)

        # base_dataclass injects .replace
        return self.replace(**updates)

    setattr(cls, "set_unpacked", set_unpacked)

    @classmethod
    def from_unpacked(cls, *, shape: tuple[int, ...] = (), **kwargs):
        """Construct an instance directly from unpacked values.

        This avoids the common but slightly wasteful pattern:
            cls.default(shape).set_unpacked(...)

        Behavior:
        - If all fields are provided in kwargs, we build the instance directly (no default allocation).
        - Otherwise we fall back to `cls.default(shape=shape)` and then apply updates (still avoids storing
          the unpacked values inside the instance).
        """
        # Direct-build path if caller provides all required fields.
        all_fields = list(field_descriptors.keys())
        has_all = all(name in kwargs for name in all_fields)
        if has_all:
            values = {}
            for name in all_fields:
                descriptor = field_descriptors[name]
                value = kwargs[name]
                if _is_packed_descriptor(descriptor):
                    values[name] = _pack_value(value, descriptor, shape)
                else:
                    values[name] = value
            return cls(**values)

        base = cls.default(shape=shape)
        updates = {}
        for name, value in kwargs.items():
            if name not in field_descriptors:
                raise KeyError(f"Unknown field '{name}' for {cls.__name__}.")
            descriptor = field_descriptors[name]
            if _is_packed_descriptor(descriptor):
                updates[name] = _pack_value(value, descriptor, shape)
            else:
                updates[name] = value
        return base.replace(**updates)

    setattr(cls, "from_unpacked", from_unpacked)
    return cls
