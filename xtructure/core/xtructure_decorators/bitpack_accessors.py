"""Decorator that adds in-memory bitpack accessors for packed fields.

Packed fields are those whose FieldDescriptor declares `packed_bits` plus
`unpacked_dtype` and `unpacked_intrinsic_shape`. The stored leaf is expected
to be a uint8 byte-stream (typically created via FieldDescriptor.packed_tensor),
and the decorator adds:

- `<field>_unpacked` property: returns logical array of shape batch + unpacked_shape
- `set_unpacked(**kwargs)` method: packs provided logical arrays and returns a new instance
"""

from __future__ import annotations

from typing import Any, Type, TypeVar

import jax
import jax.numpy as jnp

from xtructure.core.layout import get_type_layout
from xtructure.core.layout.types import PackedFieldLayout
from xtructure.io.bitpack import from_uint8, to_uint8

T = TypeVar("T")


def _unpack_value(
    packed_value: Any,
    packed_layout: PackedFieldLayout,
    batch_shape: tuple[int, ...],
):
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


def _pack_value(
    unpacked_value: Any,
    packed_layout: PackedFieldLayout,
    batch_shape: tuple[int, ...],
):
    packed_bits = packed_layout.packed_bits
    unpacked_shape = packed_layout.unpacked_intrinsic_shape

    arr = jnp.asarray(unpacked_value)
    if arr.shape[: len(batch_shape)] != batch_shape:
        raise ValueError(
            f"Unpacked value batch_shape mismatch: expected {batch_shape}, got {arr.shape[:len(batch_shape)]}."
        )
    if arr.shape[len(batch_shape) :] != unpacked_shape:
        raise ValueError(
            f"Unpacked value trailing shape mismatch: expected {unpacked_shape}, got {arr.shape[len(batch_shape):]}."
        )

    expected_packed_len = packed_layout.packed_byte_count

    flat = arr.reshape((-1,) + unpacked_shape)

    def _pack_row(row):
        return to_uint8(row, active_bits=packed_bits)

    packed_flat = jax.vmap(_pack_row)(flat)
    return packed_flat.reshape(batch_shape + (expected_packed_len,)).astype(jnp.uint8)


def add_bitpack_accessors(cls: Type[T]) -> Type[T]:
    """Attach unpacked accessors for any packed fields defined on the class."""
    type_layout = get_type_layout(cls)
    field_by_name = type_layout.field_by_name
    packed_field_layouts = type_layout.packed_field_layouts
    packed_field_layout_by_name = type_layout.packed_field_layout_by_name
    if not packed_field_layouts:
        return cls

    def _make_unpacked_property(packed_layout: PackedFieldLayout):
        def _prop(self):
            batch = getattr(self.shape, "batch", ())
            if batch == -1:
                raise TypeError(
                    f"{cls.__name__} is UNSTRUCTURED (shape.batch == -1); "
                    f"cannot infer batch for unpacking '{packed_layout.name}'."
                )
            return _unpack_value(getattr(self, packed_layout.name), packed_layout, batch)

        return property(_prop)

    for packed_layout in packed_field_layouts:
        setattr(cls, f"{packed_layout.name}_unpacked", _make_unpacked_property(packed_layout))

    def set_unpacked(self, **kwargs):
        """Return a new instance with provided packed fields updated from unpacked arrays."""
        batch = getattr(self.shape, "batch", ())
        if batch == -1:
            raise TypeError(
                f"{cls.__name__} is UNSTRUCTURED (shape.batch == -1); cannot set_unpacked."
            )

        updates = {}
        for name, value in kwargs.items():
            if name not in field_by_name:
                raise KeyError(f"Unknown field '{name}' for {cls.__name__}.")
            packed_layout = packed_field_layout_by_name.get(name)
            if packed_layout is None:
                raise TypeError(
                    f"Field '{name}' is not a packed field (missing packed_bits/unpacked metadata)."
                )
            updates[name] = _pack_value(value, packed_layout, batch)

        # base_dataclass injects .replace
        return self.replace(**updates)

    setattr(cls, "set_unpacked", set_unpacked)

    def _maybe_pack(name: str, value: Any, batch_shape: tuple[int, ...]) -> Any:
        packed_layout = packed_field_layout_by_name.get(name)
        return _pack_value(value, packed_layout, batch_shape) if packed_layout else value

    @classmethod
    def from_unpacked(cls, *, shape: tuple[int, ...] = (), **kwargs):
        """Construct an instance directly from unpacked values.

        This avoids the common but slightly wasteful pattern:
            cls.default(shape).set_unpacked(...)

        If all fields are provided we build the instance directly; otherwise we
        start from `cls.default(shape=shape)` and apply updates.
        """
        unknown = next((name for name in kwargs if name not in field_by_name), None)
        if unknown is not None:
            raise KeyError(f"Unknown field '{unknown}' for {cls.__name__}.")

        packed = {name: _maybe_pack(name, value, shape) for name, value in kwargs.items()}
        if len(packed) == len(type_layout.field_names):
            return cls(**packed)
        return cls.default(shape=shape).replace(**packed)

    setattr(cls, "from_unpacked", from_unpacked)
    return cls
