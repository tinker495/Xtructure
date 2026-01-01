from typing import Callable, Optional, Type, TypeVar

from xtructure.core.dataclass import base_dataclass
from xtructure.core.field_descriptors import get_field_descriptors
from xtructure.core.protocol import Xtructurable
from xtructure.core.type_utils import is_xtructure_dataclass_type

from .aggregate_bitpack import add_aggregate_bitpack
from .bitpack_accessors import add_bitpack_accessors
from .default import add_default_method
from .hash import hash_function_decorator
from .indexing import add_indexing_methods
from .io import add_io_methods
from .ops import add_comparison_operators
from .shape import add_shape_dtype_len
from .string_format import add_string_representation_methods
from .structure_util import add_structure_utilities
from .validation import add_runtime_validation

T = TypeVar("T")


def xtructure_dataclass(
    cls: Optional[Type[T]] = None,
    *,
    validate: bool = False,
    aggregate_bitpack: bool = False,
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
                f"bitpack must be one of 'auto','aggregate','field','off', got {bitpack!r}"
            )

        # Backward-compatible alias:
        # aggregate_bitpack=True implies at least aggregate mode.
        effective_bitpack = bitpack
        if aggregate_bitpack and bitpack == "auto":
            effective_bitpack = "aggregate"
        elif aggregate_bitpack and bitpack == "field":
            # field-only conflicts with forcing aggregate.
            raise ValueError("aggregate_bitpack=True conflicts with bitpack='field'.")
        elif aggregate_bitpack and bitpack == "off":
            raise ValueError("aggregate_bitpack=True conflicts with bitpack='off'.")

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
        descriptors = get_field_descriptors(cls)
        auto_aggregate = False
        if descriptors and effective_bitpack in ("auto",):
            # Auto aggregate if every *primitive leaf* has bits.
            def _all_bits_no_nested_arrays(t: type) -> bool:
                descs = get_field_descriptors(t)
                for fd in descs.values():
                    if is_xtructure_dataclass_type(fd.dtype):
                        # Only scalar nested supported in aggregate for now.
                        if tuple(fd.intrinsic_shape) not in ((),):
                            return False
                        if not _all_bits_no_nested_arrays(fd.dtype):
                            return False
                    else:
                        if fd.bits is None:
                            return False
                return True

            auto_aggregate = _all_bits_no_nested_arrays(cls)

        if effective_bitpack == "aggregate" or auto_aggregate:
            cls = add_aggregate_bitpack(cls)

        # add unpacked accessors / setters for in-memory packed fields
        if effective_bitpack in ("auto", "aggregate", "field"):
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
