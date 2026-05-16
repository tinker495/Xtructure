"""Decorator that adds in-memory bitpack accessors for packed fields.

Packed fields are those whose FieldDescriptor declares `packed_bits` plus a
logical unpacked shape. Type Layout / Bitpack Layout interpret that Schema
declaration into stored byte-stream facts, and the decorator adds:

- `<field>_unpacked` property: returns logical array of shape batch + unpacked_shape
- `set_unpacked(**kwargs)` method: packs provided logical arrays and returns a new instance

Field-level pack/unpack is delegated to **Bitpack Layout** (`pack_field` /
`unpack_field`) so the IO adapter and the in-memory accessor adapter share a
single source of truth.
"""

from __future__ import annotations

from typing import Any, Type, TypeVar

from xtructure.core.layout import get_type_layout
from xtructure.core.layout.bitpack import pack_field, unpack_field
from xtructure.core.layout.types import PackedFieldLayout

T = TypeVar("T")


def add_bitpack_accessors(cls: Type[T]) -> Type[T]:
    """Attach unpacked accessors for any packed fields defined on the class."""
    type_layout = get_type_layout(cls)
    packed_field_layouts = type_layout.packed_field_layouts
    if not packed_field_layouts:
        return cls

    def _make_unpacked_property(packed_layout: PackedFieldLayout):
        def _prop(self):
            batch = getattr(self.shape, "batch", ())
            if batch == -1:
                raise TypeError(
                    f"{cls.__name__} is UNSTRUCTURED (shape.batch == -1)."
                    f" Cannot infer batch for unpacking '{packed_layout.name}'."
                )
            return unpack_field(getattr(self, packed_layout.name), packed_layout, batch)

        return property(_prop)

    for packed_layout in packed_field_layouts:
        setattr(
            cls,
            f"{packed_layout.name}_unpacked",
            _make_unpacked_property(packed_layout),
        )

    def set_unpacked(self, **kwargs):
        """Return a new instance with provided packed fields updated from unpacked arrays."""
        batch = getattr(self.shape, "batch", ())
        if batch == -1:
            raise TypeError(
                f"{cls.__name__} is UNSTRUCTURED (shape.batch == -1). Cannot call set_unpacked."
            )

        updates = {}
        for name, value in kwargs.items():
            if not type_layout.has_field(name):
                raise KeyError(f"Unknown field '{name}' for {cls.__name__}.")
            packed_layout = type_layout.maybe_packed_field_layout_for(name)
            if packed_layout is None:
                raise TypeError(
                    f"Field '{name}' is not a packed field (missing packed_bits/unpacked metadata)."
                )
            updates[name] = pack_field(value, packed_layout, batch)

        return self.replace(**updates)

    setattr(cls, "set_unpacked", set_unpacked)

    def _maybe_pack(name: str, value: Any, batch_shape: tuple[int, ...]) -> Any:
        packed_layout = type_layout.maybe_packed_field_layout_for(name)
        return pack_field(value, packed_layout, batch_shape) if packed_layout else value

    @classmethod
    def from_unpacked(cls, *, shape: tuple[int, ...] = (), **kwargs):
        """Construct an instance directly from unpacked values.

        If all fields are provided we build the instance directly; otherwise we
        start from `cls.default(shape=shape)` and apply updates.
        """
        unknown = next((name for name in kwargs if not type_layout.has_field(name)), None)
        if unknown is not None:
            raise KeyError(f"Unknown field '{unknown}' for {cls.__name__}.")

        packed = {name: _maybe_pack(name, value, shape) for name, value in kwargs.items()}
        if len(packed) == len(type_layout.field_names):
            return cls(**packed)
        return cls.default(shape=shape).replace(**packed)

    setattr(cls, "from_unpacked", from_unpacked)
    return cls
