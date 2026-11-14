import builtins
from collections.abc import Callable, Iterator, Sequence
from operator import attrgetter
from types import NotImplementedType
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import chex
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Self, dataclass_transform

from .class_setup import _ClassState, install_xtructure_members
from .display import MAX_PRINT_BATCH_SIZE, SHOW_BATCH_SIZE
from .display.formatting import _custom_pretty_formatter
from .hash import _h, _h_with_uint32ed, _uint32ed, tree_to_bytes
from .indexing import _AtIndexer, _tuple_getter_values
from .layout.bitpack import pack_field
from .layout.instance_layout import get_instance_layout
from .layout.types import AdapterFieldPlan, InstanceLayout
from .structuredtype import StructuredType


@runtime_checkable
class _Indexable(Protocol):
    def __getitem__(self, index: Any) -> Any:
        ...


@dataclass_transform(eq_default=False)
class Xtructurable:
    """A structured JAX dataclass declared by inheritance.

    Common behavior lives on this base. Class creation prepares layout facts,
    compiled field updates, and optional generated packed representations.
    """

    default_shape: ClassVar[Any]
    default_dtype: ClassVar[Any]
    _xtructure_state: ClassVar[_ClassState]
    _layout_cache: InstanceLayout | None

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        if cls is Xtructurable:
            raise TypeError(
                "Xtructurable cannot be instantiated directly; subclass Xtructurable instead."
            )
        instance = super().__new__(cls)
        object.__setattr__(instance, "_layout_cache", None)
        return instance

    def __init_subclass__(
        cls,
        *,
        bitpack: Literal["auto", "aggregate", "field", "off"] = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if any(
            base is not Xtructurable and issubclass(base, Xtructurable) for base in cls.__mro__[1:]
        ):
            raise TypeError(
                f"{cls.__name__} cannot inherit from an already xtructured class; "
                "use a plain Python mixin or nested FieldDescriptor composition instead."
            )
        install_xtructure_members(cls, bitpack=bitpack)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        if self._xtructure_state.user_post_init is not None:
            self._xtructure_state.user_post_init(self, *args, **kwargs)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shared batch axes of validated data."""
        batch_shape = self.layout.batch_shape
        assert isinstance(batch_shape, tuple)
        return batch_shape

    @property
    def layout(self) -> InstanceLayout:
        if self._layout_cache is None:
            self.check_invariants()
        assert self._layout_cache is not None
        return self._layout_cache

    @property
    def dtype(self) -> Any:
        return self.layout.dtype_tuple

    @property
    def structured_type(self) -> StructuredType:
        return self.layout.structured_type

    def _map_data(self, function: Callable[..., Any], *others: Self) -> Self:
        """Map array leaves at a data boundary, validating inputs and the result."""
        self.layout
        for other in others:
            other.layout
        result = jax.tree_util.tree_map(function, self, *others)
        result.layout
        return result

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        """First batch dimension, or 1 for SINGLE instances."""
        return int(self.shape[0]) if self.shape else 1

    def __iter__(self) -> Iterator[Self]:
        """Iterate over the first batch axis; reject non-batched instances."""
        if self.shape == ():
            raise TypeError(f"Cannot iterate over SINGLE {type(self).__name__} instances.")
        return (self[index] for index in range(len(self)))

    def __getitem__(self, index: Any) -> Self:
        self.layout
        state = self._xtructure_state
        values = _tuple_getter_values(state.field_getter, self, len(state.type_layout.field_names))
        return type(self)(
            **{
                name: value[index] if isinstance(value, _Indexable) else value
                for name, value in zip(state.type_layout.field_names, values)
            }
        )

    @property
    def at(self) -> _AtIndexer[Self]:
        self.layout
        return _AtIndexer(
            self, self._xtructure_state.setter, self._xtructure_state.condition_setter
        )

    @classmethod
    def default(cls, shape: tuple[int, ...] = ()) -> Self:
        default_values: dict[str, Any] = {}

        def resolve_fill_value(field_plan: AdapterFieldPlan, field_shape: tuple[int, ...]) -> Any:
            if field_plan.fill_value_factory is not None:
                return field_plan.fill_value_factory(field_shape, field_plan.declared_dtype)
            return field_plan.fill_value

        for field_plan in cls._xtructure_state.type_layout.adapter_field_plans:
            field_shape = shape + field_plan.storage_intrinsic_shape
            if field_plan.field_kind == "primitive":
                default_values[field_plan.name] = jnp.full(
                    field_shape,
                    resolve_fill_value(field_plan, field_shape),
                    dtype=field_plan.declared_dtype,
                )
            elif field_plan.field_kind == "nested":
                nested_type = field_plan.nested_type
                assert nested_type is not None
                default_values[field_plan.name] = nested_type.default(shape=field_shape)
        return cls(**default_values)

    def reshape(self, *new_shape: int | tuple[int, ...]) -> Self:
        if len(new_shape) == 0:
            raise ValueError("new_shape must be provided")
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        else:
            new_shape = tuple(new_shape)

        if self.structured_type == StructuredType.BATCHED:
            total_length = int(np.prod(self.shape))

            # Handle -1 in new_shape by calculating the missing dimension
            new_shape_list = list(new_shape)
            if -1 in new_shape_list:
                # Count how many -1s are in the shape
                minus_one_count = new_shape_list.count(-1)
                if minus_one_count > 1:
                    raise ValueError("Only one -1 is allowed in new_shape")

                # Calculate the product of all non-negative values in new_shape
                non_negative_product = 1
                for dim in new_shape_list:
                    if isinstance(dim, (tuple, list)):
                        raise TypeError("new_shape dimensions must be integers")
                    if dim != -1:
                        non_negative_product *= dim

                # Calculate what the -1 should be
                if non_negative_product == 0:
                    raise ValueError("Cannot infer -1 dimension when other dimensions are 0")

                inferred_dim = total_length // non_negative_product
                if total_length % non_negative_product != 0:
                    raise ValueError(
                        f"Total length {total_length} is not divisible by the product of "
                        f"other dimensions {non_negative_product}"
                    )

                # Replace -1 with the calculated dimension
                minus_one_index = new_shape_list.index(-1)
                new_shape_list[minus_one_index] = inferred_dim
                new_shape = tuple(new_shape_list)

            new_total_length = np.prod(new_shape)
            batch_dim = len(self.shape)
            if total_length != new_total_length:
                raise ValueError(
                    f"Total length of the state and new shape does not match: {total_length} != {new_total_length}"
                )
            return self._map_data(lambda x: jnp.reshape(x, new_shape + x.shape[batch_dim:]))
        else:
            raise ValueError(
                f"Reshape is only supported for BATCHED structured_type. Current type: '{self.structured_type}'."
                f"Shape: {self.layout.shape_tuple}, Default Shape: {self.default_shape}"
            )

    def flatten(self) -> Self:
        if self.structured_type != StructuredType.BATCHED:
            raise ValueError(
                f"Flatten operation is only supported for BATCHED structured types. "
                f"Current type: {self.structured_type}"
            )

        current_batch_shape = self.shape
        # np.prod of an empty tuple array is 1, which is correct for total_length
        # if current_batch_shape is ().
        total_length = np.prod(np.array(current_batch_shape))
        len_current_batch_shape = len(current_batch_shape)

        return self._map_data(
            # Reshape each leaf: flatten batch dims, keep core dims.
            # core_dims are obtained by stripping batch_dims from the start of x.shape.
            lambda x: jnp.reshape(x, (total_length,) + x.shape[len_current_batch_shape:]),
        )

    def transpose(self, axes: Sequence[int] | None = None) -> Self:
        batch_ndim = len(self.shape)

        if axes is None:
            axes = tuple(range(batch_ndim - 1, -1, -1))

        def transpose_batch_only(field):
            field_ndim = field.ndim
            if field_ndim <= batch_ndim:
                return jnp.transpose(field, axes=axes)
            full_axes = list(axes) + list(range(batch_ndim, field_ndim))
            return jnp.transpose(field, axes=full_axes)

        return self._map_data(transpose_batch_only)

    def swapaxes(self, axis1: int, axis2: int) -> Self:
        batch_ndim = len(self.shape)

        axis1_norm = axis1 if axis1 >= 0 else batch_ndim + axis1
        axis2_norm = axis2 if axis2 >= 0 else batch_ndim + axis2

        if axis1_norm < 0 or axis1_norm >= batch_ndim:
            raise ValueError(f"Axis {axis1} is out of bounds for batch dimensions {self.shape}")
        if axis2_norm < 0 or axis2_norm >= batch_ndim:
            raise ValueError(f"Axis {axis2} is out of bounds for batch dimensions {self.shape}")

        return self._map_data(lambda field: jnp.swapaxes(field, axis1_norm, axis2_norm))

    @classmethod
    def random(cls, shape: tuple[int, ...] = (), key: chex.PRNGKey | None = None) -> Self:
        field_plans = cls._xtructure_state.type_layout.adapter_field_plans
        if key is None:
            key = jax.random.PRNGKey(0)

        data = {}
        keys = jax.random.split(key, len(field_plans))

        for i, field_plan in enumerate(field_plans):
            field_key = keys[i]
            field_name = field_plan.name
            target_shape = shape + field_plan.storage_intrinsic_shape

            if field_plan.random_kind == "nested":
                nested_class = field_plan.nested_type
                assert nested_class is not None
                # Recursively call random for the nested xtructure_data class.
                data[field_name] = nested_class.random(shape=target_shape, key=field_key)
            elif field_plan.random_kind == "bits_int":
                generated_bits = jax.random.bits(
                    field_key, shape=target_shape, dtype=field_plan.random_bits_dtype
                )
                if field_plan.random_view_as_signed:
                    data[field_name] = generated_bits.view(field_plan.declared_dtype)
                else:
                    data[field_name] = generated_bits
            elif field_plan.random_kind == "float":
                data[field_name] = jax.random.uniform(
                    field_key, target_shape, dtype=field_plan.random_gen_dtype
                )
            elif field_plan.random_kind == "bool":
                data[field_name] = jax.random.bernoulli(field_key, shape=target_shape)
            else:
                try:
                    data[field_name] = jnp.zeros(target_shape, dtype=field_plan.random_gen_dtype)
                except TypeError:
                    raise NotImplementedError(
                        f"Random generation for dtype {field_plan.random_gen_dtype} "
                        f"(field: {field_name}) is not implemented robustly."
                    )
        return cls(**data)

    def padding_as_batch(self, batch_shape: tuple[int, ...]) -> Self:
        if self.structured_type != StructuredType.BATCHED or len(self.shape) > 1:
            raise ValueError(
                "Padding as batch operation is only supported for BATCHED structured types "
                "with at most 1 batch dimension. "
                f"Current type: {self.structured_type}, "
                f"Current batch shape: {self.layout.batch_shape}"
            )
        if self.shape == batch_shape:
            return self

        new_default_state = self.default(batch_shape)
        new_default_state = new_default_state.at[: self.shape[0]].set(self)
        return new_default_state

    def __str__(self, **kwargs: Any) -> builtins.str:
        """Provide the base representation for user formatters calling super().__str__()."""
        state = self._xtructure_state
        if self.structured_type == StructuredType.SINGLE:
            return _custom_pretty_formatter(self, **kwargs)
        return state.renderer.render(
            self,
            max_size=kwargs.pop("max_size", MAX_PRINT_BATCH_SIZE),
            show_size=kwargs.pop("show_size", SHOW_BATCH_SIZE),
            title=f"Batched {type(self).__name__}",
            **kwargs,
        )

    def str(self, **kwargs: Any) -> builtins.str:
        if self.structured_type == StructuredType.SINGLE:
            return self._xtructure_state.single_formatter(self, **kwargs)
        return Xtructurable.__str__(self, **kwargs)

    bytes = property(tree_to_bytes)
    uint32ed = property(_uint32ed)
    hash = _h
    hash_with_uint32ed = _h_with_uint32ed

    # Array equality intentionally differs from object.__eq__'s bool annotation.
    # pyrefly: ignore [bad-override]
    def __eq__(self, other: Any) -> jax.Array | NotImplementedType:
        self.layout
        if not isinstance(other, self.__class__):
            return NotImplemented
        other.layout

        tree_equal = jax.tree_util.tree_map(lambda x, y: jnp.all(x == y), self, other)
        return jax.tree_util.tree_reduce(jnp.logical_and, tree_equal)

    # Array inequality intentionally differs from object.__ne__'s bool annotation.
    # pyrefly: ignore [bad-override]
    def __ne__(self, other: Any) -> jax.Array | NotImplementedType:
        self.layout
        if not isinstance(other, self.__class__):
            return NotImplemented
        other.layout

        tree_equal = jax.tree_util.tree_map(lambda x, y: jnp.any(x != y), self, other)
        return jax.tree_util.tree_reduce(jnp.logical_or, tree_equal)

    def save(self, path: builtins.str, *, packed: builtins.bool = True) -> None:
        """Saves the instance to a .npz file."""
        from xtructure.io import save

        return save(path, self, packed=packed)

    @classmethod
    def load(cls, path: builtins.str) -> Self:
        """Loads an instance from a .npz file."""
        from xtructure.io import load

        loaded_instance = load(path)
        if not isinstance(loaded_instance, cls):
            raise TypeError(
                f"Loaded instance is of type {type(loaded_instance).__name__}, "
                f"but expected {cls.__name__}."
            )
        return loaded_instance

    def _validate_array_field(self, field_plan: AdapterFieldPlan, value: Any) -> None:
        try:
            # Array metadata is enough; inspecting host data must not transfer it.
            array = (
                value
                if isinstance(value, (jax.Array, np.ndarray, np.generic))
                else jnp.asarray(value)
            )
        except TypeError as err:
            raise TypeError(
                f"{type(self).__name__}.{field_plan.name} could not be coerced to a JAX array for validation."
            ) from err

        try:
            expected_dtype = jnp.dtype(field_plan.declared_dtype)
        except TypeError as err:
            raise TypeError(
                f"{type(self).__name__}.{field_plan.name} has unsupported dtype "
                f"{field_plan.declared_dtype}."
            ) from err

        if array.dtype != expected_dtype:
            raise TypeError(
                f"{type(self).__name__}.{field_plan.name} expected dtype {expected_dtype}, "
                f"got {array.dtype}."
            )

        intrinsic_shape = tuple(field_plan.storage_intrinsic_shape)
        if intrinsic_shape:
            intrinsic_len = len(intrinsic_shape)
            if array.ndim < intrinsic_len:
                raise ValueError(
                    f"{type(self).__name__}.{field_plan.name} expected trailing shape {intrinsic_shape}, "
                    f"but array has shape {array.shape}."
                )
            trailing_shape = array.shape[-intrinsic_len:]
            if tuple(trailing_shape) != intrinsic_shape:
                raise ValueError(
                    f"{type(self).__name__}.{field_plan.name} expected trailing shape {intrinsic_shape}, "
                    f"but got {trailing_shape}."
                )

    def check_invariants(self) -> None:
        """Validate actual fields and cache their layout after all checks pass."""
        object.__setattr__(self, "_layout_cache", None)
        for field_plan in self._xtructure_state.type_layout.adapter_field_plans:
            value = attrgetter(field_plan.name)(self)

            if field_plan.field_kind == "nested":
                nested_type = field_plan.nested_type
                assert nested_type is not None
                if not isinstance(value, nested_type):
                    raise TypeError(
                        f"{type(self).__name__}.{field_plan.name} expected instance of "
                        f"{nested_type.__name__}, got {type(value).__name__}."
                    )
                value.check_invariants()
            else:
                self._validate_array_field(field_plan, value)

        layout = get_instance_layout(self)
        if layout.structured_type == StructuredType.UNSTRUCTURED:
            raise ValueError(f"{type(self).__name__}: {layout.mismatch_reason}")
        for field_plan in self._xtructure_state.type_layout.adapter_field_plans:
            if field_plan.validator is not None:
                field_plan.validator(attrgetter(field_plan.name)(self))
        object.__setattr__(self, "_layout_cache", layout)

    def replace(self, **kwargs: Any) -> Self:
        self.layout
        type_layout = self._xtructure_state.type_layout
        unexpected = tuple(name for name in kwargs if not type_layout.has_field(name))
        if unexpected:
            raise TypeError(
                f"{type(self).__name__}.replace got unexpected field(s): "
                + ", ".join(repr(name) for name in unexpected)
            )
        return type(self)(
            **{
                name: kwargs[name] if name in kwargs else attrgetter(name)(self)
                for name in type_layout.field_names
            }
        )

    def set_unpacked(self, **kwargs: Any) -> Self:
        """Return a new instance with provided packed fields updated from unpacked arrays."""
        type_layout = self._xtructure_state.type_layout
        updates = {}
        for name, value in kwargs.items():
            if not type_layout.has_field(name):
                raise KeyError(f"Unknown field '{name}' for {type(self).__name__}.")
            packed_layout = type_layout.maybe_packed_field_layout_for(name)
            if packed_layout is None:
                raise TypeError(
                    f"Field '{name}' is not a packed field (missing packed_bits/unpacked metadata)."
                )
            updates[name] = pack_field(value, packed_layout, self.shape)

        return self.replace(**updates)

    @classmethod
    def from_unpacked(cls, *, shape: tuple[int, ...] = (), **kwargs: Any) -> Self:
        """Construct an instance directly from unpacked values.

        If all fields are provided we build the instance directly; otherwise we
        start from `cls.default(shape=shape)` and apply updates.
        """
        type_layout = cls._xtructure_state.type_layout
        unknown = next((name for name in kwargs if not type_layout.has_field(name)), None)
        if unknown is not None:
            raise KeyError(f"Unknown field '{unknown}' for {cls.__name__}.")

        packed = {}
        for name, value in kwargs.items():
            layout = type_layout.maybe_packed_field_layout_for(name)
            packed[name] = pack_field(value, layout, shape) if layout is not None else value
        if len(packed) == len(type_layout.field_names):
            return cls(**packed)
        return cls.default(shape=shape).replace(**packed)
