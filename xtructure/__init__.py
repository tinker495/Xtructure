from . import io, xtructure_numpy
from .bgpq import BGPQ
from .core import (
    FieldDescriptor,
    StructuredType,
    Xtructurable,
    base_dataclass,
    broadcast_intrinsic_shape,
    clone_field_descriptor,
    descriptor_metadata,
    with_intrinsic_shape,
    xtructure_dataclass,
)
from .hashtable import HashIdx, HashTable
from .queue import Queue
from .stack import Stack

# Alias xtructure_numpy as numpy for cleaner imports (e.g., from xtructure import numpy as xnp)
numpy = xtructure_numpy

__all__ = [
    # bgpq.py
    "BGPQ",
    # hash.py
    "HashTable",
    "HashIdx",
    # queue.py
    "Queue",
    # stack.py
    "Stack",
    # core.dataclass.py
    "Xtructurable",
    "base_dataclass",
    "xtructure_dataclass",
    "StructuredType",
    # core.field_descriptors.py
    "FieldDescriptor",
    "clone_field_descriptor",
    "with_intrinsic_shape",
    "broadcast_intrinsic_shape",
    "descriptor_metadata",
    # numpy alias
    "numpy",
    # xtructure_numpy.py (top-level xnp module)
    "xtructure_numpy",
    # io.py
    "io",
]
