"""Public API for xtructure.

Imports are resolved lazily so internal modules can depend on concrete owners
without re-entering the top-level facade during package initialization.
"""

from __future__ import annotations

from xtructure._lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "BGPQ",
    "HashTable",
    "HashIdx",
    "Queue",
    "Stack",
    "Xtructurable",
    "base_dataclass",
    "xtructure_dataclass",
    "StructuredType",
    "FieldDescriptor",
    "clone_field_descriptor",
    "numpy",
    "io",
]

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "BGPQ": (".bgpq.bgpq", "BGPQ"),
    "HashTable": (".hashtable.table", "HashTable"),
    "HashIdx": (".hashtable.types", "HashIdx"),
    "Queue": (".queue.queue", "Queue"),
    "Stack": (".stack.stack", "Stack"),
    "Xtructurable": (".core.protocol", "Xtructurable"),
    "base_dataclass": (".core.dataclass", "base_dataclass"),
    "xtructure_dataclass": (".core.xtructure_decorators", "xtructure_dataclass"),
    "StructuredType": (".core.structuredtype", "StructuredType"),
    "FieldDescriptor": (".core.field_descriptors", "FieldDescriptor"),
    "clone_field_descriptor": (".core.field_descriptor_utils", "clone_field_descriptor"),
    "numpy": (".numpy", None),
    "io": (".io", None),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
