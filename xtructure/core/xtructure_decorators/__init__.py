from typing import Callable, Optional, Type, TypeVar

from xtructure.core.dataclass import base_dataclass
from xtructure.core.layout import get_type_layout
from xtructure.core.protocol import Xtructurable

from .layout_adapters.aggregate_bitpack import add_aggregate_bitpack
from .layout_adapters.bitpack_accessors import add_bitpack_accessors
from .layout_adapters.default import add_default_method
from .layout_adapters.indexing import add_indexing_methods
from .layout_adapters.shape import add_shape_dtype_len
from .layout_adapters.structure_util import add_structure_utilities
from .layout_adapters.validation import add_runtime_validation
from .pytree_adapters.hash import hash_function_decorator
from .pytree_adapters.io import add_io_methods
from .pytree_adapters.ops import add_comparison_operators
from .pytree_adapters.string_format import add_string_representation_methods

T = TypeVar("T")


def xtructure_dataclass(
    cls: Optional[Type[T]] = None,
    *,
    validate: bool = False,
    bitpack: str = "auto",
) -> Callable[[Type[T]], Type[Xtructurable[T]]] | Type[Xtructurable[T]]:
    """
    Decorator that ensures the input class is a `base_dataclass` (or converts
    it to one) and then augments it with additional functionality related to its
    structure, type, and operations like indexing, default instance creation,
    random instance generation, and string representation.

    It adds properties like `shape`, `dtype`, `default_shape`, `structured_type`,
    `batch_shape`, and methods like `__getitem__`, `__len__`, `reshape`,
    `flatten`, `transpose`, `random`, and `__str__`.

    Args:
        cls: The class to be decorated. It is expected to have a `default`
             classmethod for some functionalities.
        validate: When True, injects a runtime validator that checks field
            dtypes and trailing shapes after every instantiation.

    Returns:
        The decorated class with the aforementioned additional functionalities.
    """

    def _decorate(target_cls: Type[T]) -> Type[Xtructurable[T]]:
        if bitpack not in ("auto", "aggregate", "field", "off"):
            raise ValueError(
                f"bitpack must be one of 'auto', 'aggregate', 'field', 'off', got {bitpack!r}"
            )

        cls = base_dataclass(target_cls)

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

        # Bitpacking policy:
        # - off: no in-memory bitpack helpers
        # - field: only packed_tensor accessors
        # - aggregate: force aggregate packing for the whole dataclass
        # - auto: enable aggregate if all primitive leaves declare bits (nested supported for scalar nesting)
        type_layout = get_type_layout(cls)
        auto_aggregate = (
            bool(type_layout.fields)
            and bitpack == "auto"
            and type_layout.aggregate_bitpack.eligible
        )

        if bitpack == "aggregate" or auto_aggregate:
            cls = add_aggregate_bitpack(cls)

        # add unpacked accessors / setters for in-memory packed fields
        if bitpack in ("auto", "aggregate", "field"):
            cls = add_bitpack_accessors(cls)

        # add string representation methods
        cls = add_string_representation_methods(cls)

        # add hash function
        cls = hash_function_decorator(cls)

        # add comparison operators
        cls = add_comparison_operators(cls)

        # add io methods
        cls = add_io_methods(cls)

        # add runtime validation if requested
        cls = add_runtime_validation(cls, enabled=validate)

        setattr(cls, "is_xtructed", True)

        return cls

    if cls is None:
        return _decorate
    return _decorate(cls)
