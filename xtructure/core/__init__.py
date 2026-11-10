"""Core public facade for xtructure."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xtructure._lazy_imports import lazy_dir, load_lazy_export

if TYPE_CHECKING:
    from . import xtructure_numpy as xtructure_numpy
    from .dataclass import base_dataclass as base_dataclass
    from .field_descriptor_utils import clone_field_descriptor as clone_field_descriptor
    from .field_descriptors import FieldDescriptor as FieldDescriptor
    from .protocol import Xtructurable as Xtructurable
    from .structuredtype import StructuredType as StructuredType

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "Xtructurable": (".protocol", "Xtructurable"),
    "StructuredType": (".structuredtype", "StructuredType"),
    "base_dataclass": (".dataclass", "base_dataclass"),
    "FieldDescriptor": (".field_descriptors", "FieldDescriptor"),
    "clone_field_descriptor": (".field_descriptor_utils", "clone_field_descriptor"),
    "xtructure_numpy": (".xtructure_numpy", None),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
