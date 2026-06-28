"""Core public facade for xtructure.

Concrete modules inside ``xtructure.core`` import each other directly; this
facade remains lazy for external compatibility.
"""

from __future__ import annotations

from xtructure._lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "Xtructurable",
    "StructuredType",
    "base_dataclass",
    "xtructure_dataclass",
    "FieldDescriptor",
    "clone_field_descriptor",
    "with_intrinsic_shape",
    "broadcast_intrinsic_shape",
    "descriptor_metadata",
    "xtructure_numpy",
]

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "Xtructurable": (".protocol", "Xtructurable"),
    "StructuredType": (".structuredtype", "StructuredType"),
    "base_dataclass": (".dataclass", "base_dataclass"),
    "xtructure_dataclass": (".xtructure_decorators", "xtructure_dataclass"),
    "FieldDescriptor": (".field_descriptors", "FieldDescriptor"),
    "clone_field_descriptor": (".field_descriptor_utils", "clone_field_descriptor"),
    "with_intrinsic_shape": (".field_descriptor_utils", "with_intrinsic_shape"),
    "broadcast_intrinsic_shape": (".field_descriptor_utils", "broadcast_intrinsic_shape"),
    "descriptor_metadata": (".field_descriptor_utils", "descriptor_metadata"),
    "xtructure_numpy": (".xtructure_numpy", None),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
