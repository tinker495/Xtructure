"""Decorator that adds in-memory bitpack accessors for packed fields.

Packed fields are those whose FieldDescriptor declares `packed_bits` plus a
logical unpacked shape. Type Layout / Bitpack Layout interpret that Schema
declaration into stored byte-stream facts, and class preparation adds:

- `<field>_unpacked` property: returns logical array of shape batch + unpacked_shape
Common construction and update methods are inherited from Xtructurable.

Field-level pack/unpack is delegated to **Bitpack Layout** (`pack_field` /
`unpack_field`) so the IO adapter and the in-memory accessor adapter share a
single source of truth.
"""

from __future__ import annotations

from operator import attrgetter
from typing import Type, TypeVar

from xtructure.core.layout import get_type_layout
from xtructure.core.layout.bitpack import unpack_field
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
            return unpack_field(attrgetter(packed_layout.name)(self), packed_layout, self.shape)

        return property(_prop)

    for packed_layout in packed_field_layouts:
        setattr(
            cls,
            f"{packed_layout.name}_unpacked",
            _make_unpacked_property(packed_layout),
        )

    return cls
