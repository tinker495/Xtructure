"""Core public facade for xtructure."""

from . import xtructure_numpy
from .dataclass import base_dataclass
from .field_descriptor_utils import clone_field_descriptor
from .field_descriptors import FieldDescriptor
from .protocol import Xtructurable
from .structuredtype import StructuredType

__all__ = [
    "Xtructurable",
    "StructuredType",
    "base_dataclass",
    "FieldDescriptor",
    "clone_field_descriptor",
    "xtructure_numpy",
]
