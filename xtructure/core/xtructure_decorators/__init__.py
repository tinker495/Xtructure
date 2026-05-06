import functools
from typing import Callable, Optional, Type, TypeVar

from xtructure.core.dataclass import base_dataclass
from xtructure.core.layout import get_instance_layout, get_type_layout
from xtructure.core.layout.instance_layout import (
    primitive_value_dtype,
    primitive_value_shape,
)
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


def _inject_layout_cache_post_init(target_cls: Type[T]) -> Type[T]:
    """Ensure every xtructure instance owns a hashable Layout Cache."""
    original_post_init = getattr(target_cls, "__post_init__", None)

    if getattr(original_post_init, "__xtructure_layout_cache_hook__", False):
        return target_cls

    if original_post_init is None:

        def _layout_cache_post_init(self, *args, **kwargs):
            del args, kwargs
            object.__setattr__(self, "_layout_cache", get_instance_layout(self))

    else:

        @functools.wraps(original_post_init)
        def _layout_cache_post_init(self, *args, **kwargs):
            original_post_init(self, *args, **kwargs)
            object.__setattr__(self, "_layout_cache", get_instance_layout(self))

    setattr(_layout_cache_post_init, "__xtructure_layout_cache_hook__", True)
    setattr(_layout_cache_post_init, "__xtructure_user_post_init__", original_post_init)
    setattr(target_cls, "__post_init__", _layout_cache_post_init)
    return target_cls


def _expected_raw_shape(batch_shape: tuple[int, ...], intrinsic_shape: tuple[int, ...]):
    return tuple(batch_shape) + tuple(intrinsic_shape)


def _can_reuse_layout_cache(old_layout, type_layout, instance, changed_names: set[str]) -> bool:
    if old_layout is None or old_layout.batch_shape == -1:
        return False

    for name in changed_names:
        field_layout = type_layout.field_for(name)
        if field_layout.is_nested:
            return False

        value = getattr(instance, name)
        if primitive_value_shape(value) != _expected_raw_shape(
            old_layout.batch_shape, field_layout.intrinsic_shape
        ):
            return False
        if primitive_value_dtype(value) != getattr(old_layout.dtype_tuple, name):
            return False

    return True


def _install_layout_cache_replace(cls: Type[T], *, validate: bool) -> Type[T]:
    """Install a Layout Cache-aware replace fast path for xtructure dataclasses."""
    type_layout = get_type_layout(cls)
    field_names = type_layout.field_names
    field_name_set = frozenset(field_names)
    post_init = getattr(cls, "__post_init__", None)
    user_post_init = getattr(post_init, "__xtructure_user_post_init__", None)

    def _layout_cache_replace(self, **kwargs):
        unexpected = tuple(name for name in kwargs if name not in field_name_set)
        if unexpected:
            unexpected_names = ", ".join(repr(name) for name in unexpected)
            raise TypeError(f"{cls.__name__}.replace got unexpected field(s): {unexpected_names}")

        old_layout = getattr(self, "_layout_cache", None)
        changed_names = set(kwargs)
        replacement = cls.__new__(cls)
        for name in field_names:
            object.__setattr__(replacement, name, kwargs.get(name, getattr(self, name)))

        if user_post_init is not None:
            user_post_init(replacement)

        if _can_reuse_layout_cache(old_layout, type_layout, replacement, changed_names):
            object.__setattr__(replacement, "_layout_cache", old_layout)
        else:
            object.__setattr__(replacement, "_layout_cache", get_instance_layout(replacement))

        if validate:
            replacement.check_invariants()
        return replacement

    setattr(cls, "replace", _layout_cache_replace)
    return cls


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
        bitpack: Bitpacking policy: ``"auto"``, ``"aggregate"``, ``"field"``,
            or ``"off"``.

    Returns:
        The decorated class with the aforementioned additional functionalities.
    """

    def _decorate(target_cls: Type[T]) -> Type[Xtructurable[T]]:
        if bitpack not in ("auto", "aggregate", "field", "off"):
            raise ValueError(
                f"bitpack must be one of 'auto', 'aggregate', 'field', 'off', got {bitpack!r}"
            )

        target_cls = _inject_layout_cache_post_init(target_cls)
        cls = base_dataclass(target_cls, static_fields=("_layout_cache",))

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

        # Keep replace() from paying a fresh Layout Cache build when the
        # replacement preserves primitive field shape/dtype signatures.
        cls = _install_layout_cache_replace(cls, validate=validate)

        setattr(cls, "is_xtructed", True)

        return cls

    if cls is None:
        return _decorate
    return _decorate(cls)
