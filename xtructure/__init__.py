"""Public API for xtructure."""

from __future__ import annotations

from . import io, numpy
from .bgpq.bgpq import BGPQ
from .core.dataclass import base_dataclass
from .core.field_descriptor_utils import (
    broadcast_intrinsic_shape,
    clone_field_descriptor,
    descriptor_metadata,
    with_intrinsic_shape,
)
from .core.field_descriptors import FieldDescriptor
from .core.protocol import Xtructurable
from .core.structuredtype import StructuredType
from .core.xtructure_decorators import xtructure_dataclass
from .hashtable.table import HashTable
from .hashtable.types import HashIdx
from .queue.queue import Queue
from .stack.stack import Stack

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
    "with_intrinsic_shape",
    "broadcast_intrinsic_shape",
    "descriptor_metadata",
    "numpy",
    "io",
]
