from typing import Any, ClassVar, Protocol
from typing import Tuple as TypingTuple
from typing import Type, TypeVar, runtime_checkable

import chex

from .structuredtype import StructuredType

T = TypeVar("T")


# Protocol defining the interface added by @xtructure_data
class _XtructurableMeta(type(Protocol)):
    def __instancecheck__(cls, instance) -> bool:
        return bool(getattr(instance, "is_xtructed", False))

    def __subclasscheck__(cls, subclass) -> bool:
        return bool(getattr(subclass, "is_xtructed", False))


@runtime_checkable
class Xtructurable(Protocol[T], metaclass=_XtructurableMeta):
    # A flag to indicate that the class has been processed by the xtructure_dataclass decorator.
    is_xtructed: ClassVar[bool]

    # The default shape of the structure, calculated at class creation time.
    # This is a namedtuple whose fields mirror the class attributes.
    default_shape: ClassVar[Any]

    # The default dtype of the structure, calculated at class creation time.
    # This is a namedtuple whose fields mirror the class attributes.
    default_dtype: ClassVar[Any]

    # Methods and properties added by add_shape_dtype_len
    @property
    def shape(self) -> Any:
        """The shape of the data in the object, as a dynamically-generated namedtuple."""
        ...

    @property
    def dtype(self) -> Any:
        """The dtype of the data in the object, as a dynamically-generated namedtuple."""
        ...

    @property
    def ndim(self) -> int:
        """Number of batch dimensions for structured instances."""
        ...

    # Method added by add_indexing_methods (responsible for __getitem__)
    def __getitem__(self: T, index: Any) -> T:
        ...

    # Method added by add_shape_dtype_len
    def __len__(self) -> int:
        ...

    # Methods and properties added by add_structure_utilities
    # Assumes the class T has a 'default' classmethod as per the decorator's assertion
    @classmethod
    def default(cls: Type[T], shape: Any = ...) -> T:
        ...

    @property
    def default_shape(self) -> Any:  # Derived from self.default().shape
        ...

    @property
    def structured_type(self) -> "StructuredType":  # Forward reference for StructuredType
        ...

    @property
    def batch_shape(self) -> TypingTuple[int, ...] | int:
        ...

    def reshape(self: T, *new_shape: int | TypingTuple[int, ...]) -> T:
        ...

    def flatten(self: T) -> T:
        ...

    def transpose(self: T, axes: TypingTuple[int, ...] | None = ...) -> T:
        ...

    @classmethod
    def random(
        cls: Type[T], shape: TypingTuple[int, ...] = ..., key: chex.PRNGKey = ...
    ) -> T:  # Ellipsis for default value
        ...

    # Methods and properties added by add_string_representation_methods
    def __str__(
        self,
    ) -> str:  # The actual implementation takes **kwargs, but signature can be simpler for Protocol
        ...

    def str(self) -> str:  # Alias for __str__
        ...

    # Method added by add_indexing_methods
    @property
    def at(self: T) -> "AtIndexer":
        ...

    @property
    def bytes(self: T) -> chex.Array:
        ...

    @property
    def uint32ed(self: T) -> chex.Array:
        ...

    def hash(self: T, seed: int = 0) -> int:
        ...

    def hash_with_uint32ed(self: T, seed: int = 0) -> TypingTuple[int, chex.Array]:
        ...

    def hash_pair(self: T, seed: int = 0) -> TypingTuple[int, int]:
        ...

    def hash_pair_with_uint32ed(
        self: T, seed: int = 0
    ) -> TypingTuple[TypingTuple[int, int], chex.Array]:
        ...

    # Method added by add_comparison_operators
    def __eq__(self, other: Any) -> bool:
        ...

    def __ne__(self, other: Any) -> bool:
        ...

    # Methods added by add_io_methods
    def save(self: T, path: str) -> None:
        ...

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        ...

    # Method added by add_runtime_validation
    def check_invariants(self) -> None:
        ...

    # Methods added by base_dataclass
    @classmethod
    def from_tuple(cls: Type[T], args: TypingTuple[Any, ...]) -> T:
        ...

    def to_tuple(self) -> TypingTuple[Any, ...]:
        ...

    def replace(self: T, **kwargs: Any) -> T:
        ...


class AtIndexer(Protocol[T]):
    def __getitem__(self: T, index: Any) -> "Updater":
        ...


class Updater(Protocol[T]):
    def set(self: T, value: Any) -> T:
        ...

    def set_as_condition(self: T, condition: chex.Array, value: Any) -> T:
        ...
