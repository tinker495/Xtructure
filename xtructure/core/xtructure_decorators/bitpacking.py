from __future__ import annotations

from typing import Type, TypeVar

from xtructure.core.bitpacking import PackedXtructure, pack_instance, unpack_instance

T = TypeVar("T")


def add_bitpacking_methods(cls: Type[T]) -> Type[T]:
    """Augment an @xtructure_dataclass with packed (byte) serialization helpers."""

    def to_packed(self: T, *, validate_range: bool = True) -> PackedXtructure:
        return pack_instance(self, validate_range=validate_range)

    @classmethod
    def from_packed(cls_: Type[T], packed: PackedXtructure) -> T:
        return unpack_instance(cls_, packed)

    setattr(cls, "to_packed", to_packed)
    setattr(cls, "from_packed", from_packed)
    return cls

