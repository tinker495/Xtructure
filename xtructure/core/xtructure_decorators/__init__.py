import functools
import importlib
import warnings
from typing import Callable, Optional, Type, TypeVar

from typing_extensions import dataclass_transform

from xtructure.core.dataclass import base_dataclass
from xtructure.core.layout.instance_layout import (
    get_instance_layout,
    primitive_value_dtype,
    primitive_value_shape,
)
from xtructure.core.layout.type_layout import get_type_layout
from xtructure.core.protocol import Xtructurable

T = TypeVar("T")
X = TypeVar("X", bound=Xtructurable)


def _adapter(module_name: str, attr_name: str):
    return getattr(importlib.import_module(module_name, __name__), attr_name)


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
            old_layout.batch_shape,
            type_layout.storage_intrinsic_shape_for(field_layout),
        ):
            return False
        if primitive_value_dtype(value) != getattr(old_layout.dtype_tuple, name):
            return False

    return True


def _install_layout_cache_replace(cls: Type[X], *, validate: bool) -> Type[X]:
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
            raise TypeError(
                f"{type_layout.cls.__name__}.replace got unexpected field(s): {unexpected_names}"
            )

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


def install_xtructure_members(
    target_cls: Type[T],
    *,
    validate: bool = False,
    bitpack: str = "auto",
) -> Type[T]:
    """Apply the shared transformation for inherited and decorator-built classes."""
    if bitpack not in ("auto", "aggregate", "field", "off"):
        raise ValueError(
            f"bitpack must be one of 'auto', 'aggregate', 'field', 'off', got {bitpack!r}"
        )

    target_cls = _inject_layout_cache_post_init(target_cls)
    cls = base_dataclass(target_cls, static_fields=("_layout_cache",))
    cls = _adapter(".layout_adapters.default", "add_default_method")(cls)
    if not hasattr(cls, "default"):
        raise RuntimeError("xtructureclass must have a default method.")

    cls = _adapter(".layout_adapters.shape", "add_shape_dtype_len")(cls)
    cls = _adapter(".layout_adapters.indexing", "add_indexing_methods")(cls)
    cls = _adapter(".layout_adapters.structure_util", "add_structure_utilities")(cls)

    type_layout = get_type_layout(cls)
    auto_aggregate = (
        bool(type_layout.fields) and bitpack == "auto" and type_layout.aggregate_bitpack.eligible
    )
    if bitpack == "aggregate" or auto_aggregate:
        cls = _adapter(".layout_adapters.aggregate_bitpack", "add_aggregate_bitpack")(
            cls,
            install_xtructure_members,
        )

    if bitpack in ("auto", "aggregate", "field"):
        cls = _adapter(".layout_adapters.bitpack_accessors", "add_bitpack_accessors")(cls)

    cls = _adapter(".pytree_adapters.string_format", "add_string_representation_methods")(cls)
    cls = _adapter(".pytree_adapters.hash", "hash_function_decorator")(cls)
    cls = _adapter(".pytree_adapters.ops", "add_comparison_operators")(cls)
    cls = _adapter(".pytree_adapters.io", "add_io_methods")(cls)
    cls = _adapter(".layout_adapters.validation", "add_runtime_validation")(
        cls,
        enabled=validate,
    )

    # Reuse Layout Cache when replacement preserves primitive shape/dtype signatures.
    cls = _install_layout_cache_replace(cls, validate=validate)

    # TODO(#43): temporary — accepts decorator-built classes; delete with the decorator.
    setattr(cls, "is_xtructed", True)
    return cls


@dataclass_transform()
def xtructure_dataclass(
    cls: Optional[Type[T]] = None,
    *,
    validate: bool = False,
    bitpack: str = "auto",
) -> Callable[[Type[T]], Type[T]] | Type[T]:
    """
    Legacy compatibility decorator for Xtructure data structures.

    .. warning::
        Applying this decorator emits :class:`FutureWarning`. Define new classes
        with ``class MyData(Xtructurable, validate=False, bitpack="auto"):``
        instead, using ``from xtructure import Xtructurable``.
        Move any decorator options into the class header and remove the decorator.
        ``python -W error::FutureWarning`` rejects legacy decoration before the
        class is transformed. The ``base_dataclass`` decorator is unaffected.

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

    def _decorate(target_cls: Type[T]) -> Type[T]:
        if Xtructurable in target_cls.__mro__ or getattr(target_cls, "is_xtructed", False):
            raise TypeError(
                f"{target_cls.__name__} is already xtructured; "
                "do not mix Xtructurable inheritance with legacy @xtructure_dataclass "
                "or decorate twice."
            )
        warnings.warn(
            f"@xtructure_dataclass on {target_cls.__name__} is legacy and deprecated. "
            f"Use class {target_cls.__name__}(Xtructurable, "
            f"validate={validate!r}, bitpack={bitpack!r}): instead; "
            "import it with 'from xtructure import Xtructurable'.",
            FutureWarning,
            stacklevel=2 if cls is None else 3,
        )
        return install_xtructure_members(target_cls, validate=validate, bitpack=bitpack)

    if cls is None:
        return _decorate
    return _decorate(cls)
