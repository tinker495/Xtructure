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
import numpy as np

from xtructure.core.field_descriptors import FieldDescriptor, get_field_descriptors
from xtructure.io.bitpack import from_uint8, packed_num_bytes, to_uint8

T = TypeVar("T")


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

    flat = arr.reshape((-1,) + unpacked_shape)

    def _pack_row(row):
        return to_uint8(row, active_bits=packed_bits)

    packed_flat = jax.vmap(_pack_row)(flat)
    return packed_flat.reshape(batch_shape + (expected_packed_len,)).astype(jnp.uint8)


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
