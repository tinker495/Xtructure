"""Public API for xtructure.

Imports are resolved lazily so internal modules can depend on concrete owners
without re-entering the top-level facade during package initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import numpy as numpy
    from . import io as io
    from .bgpq.bgpq import BGPQ as BGPQ
    from .core.dataclass import base_dataclass as base_dataclass
    from .core.field_descriptor_utils import clone_field_descriptor as clone_field_descriptor
    from .core.field_descriptors import FieldDescriptor as FieldDescriptor
    from .core.protocol import Xtructurable as Xtructurable
    from .core.structuredtype import StructuredType as StructuredType
    from .core.xtructure_decorators import xtructure_dataclass as xtructure_dataclass
    from .hashtable.table import HashTable as HashTable
    from .hashtable.types import (
        HashIdx as HashIdx,
        HashTableProbe as HashTableProbe,
    )
    from .queue.queue import Queue as Queue
    from .stack.stack import Stack as Stack

from xtructure._lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "BGPQ",
    "HashTable",
    "HashIdx",
    "HashTableProbe",
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
    "HashTableProbe": (".hashtable.types", "HashTableProbe"),
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
