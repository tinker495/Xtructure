"""Core public facade for xtructure.

Concrete modules inside ``xtructure.core`` import each other directly; this
facade remains lazy for external compatibility.
"""

from __future__ import annotations

import importlib
from typing import Any

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
    "get_type_layout",
    "get_instance_layout",
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
    "get_type_layout": (".layout.type_layout", "get_type_layout"),
    "get_instance_layout": (".layout.instance_layout", "get_instance_layout"),
    "xtructure_numpy": (".xtructure_numpy", None),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name, __name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
