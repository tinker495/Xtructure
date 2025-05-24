from typing import Type, TypeVar

import chex

from xtructure.core.protocol import Xtructurable

from .default import add_default_method
from .hash import hash_function_decorator
from .indexing import add_indexing_methods
from .ops import add_comparison_operators
from .shape import add_shape_dtype_len
from .string_format import add_string_representation_methods
from .structure_util import add_structure_utilities

T = TypeVar("T")


def xtructure_dataclass(cls: Type[T]) -> Type[Xtructurable[T]]:
    """
    Decorator that ensures the input class is a `chex.dataclass` (or converts
    it to one) and then augments it with additional functionality related to its
    structure, type, and operations like indexing, default instance creation,
    random instance generation, and string representation.

    It adds properties like `shape`, `dtype`, `default_shape`, `structured_type`,
    `batch_shape`, and methods like `__getitem__`, `__len__`, `reshape`,
    `flatten`, `random`, and `__str__`.

    Args:
        cls: The class to be decorated. It is expected to have a `default`
             classmethod for some functionalities.

    Returns:
        The decorated class with the aforementioned additional functionalities.
    """
    cls = chex.dataclass(cls)

    # Ensure class has a default method for initialization
    cls = add_default_method(cls)

    # Ensure class has a default method for initialization
    assert hasattr(cls, "default"), "xtructureclass must have a default method."

    # add shape and dtype and len
    cls = add_shape_dtype_len(cls)

    # add indexing methods
    cls = add_indexing_methods(cls)

    # add structure utilities and random
    cls = add_structure_utilities(cls)

    # add string representation methods
    cls = add_string_representation_methods(cls)

    # add hash function
    cls = hash_function_decorator(cls)

    # add comparison operators
    cls = add_comparison_operators(cls)

    setattr(cls, "is_xtructed", True)

    return cls
