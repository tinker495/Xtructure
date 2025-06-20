from typing import Any, ClassVar, Dict, NamedTuple, Protocol
from typing import Tuple as TypingTuple
from typing import Type, TypeVar

import chex

from .structuredtype import StructuredType

T = TypeVar("T")


class shape_tuple(NamedTuple):
    batch: tuple[int, ...]
    fields: Dict[str, Any]


class dtype_tuple(NamedTuple):
    fields: Dict[str, Any]


# Protocol defining the interface added by @xtructure_data
class Xtructurable(Protocol[T]):
    # A flag to indicate that the class has been processed by the xtructure_dataclass decorator.
    is_xtructed: ClassVar[bool]

    # The default shape of the structure, calculated at class creation time.
    # This is a namedtuple whose fields mirror the class attributes.
    default_shape: ClassVar[Any]

    # The default dtype of the structure, calculated at class creation time.
    # This is a namedtuple whose fields mirror the class attributes.
    default_dtype: ClassVar[Any]

    # Fields from the original class that chex.dataclass would process
    # These are implicitly part of T. For the protocol to be complete,
    # it assumes T will have __annotations__.
    __annotations__: Dict[str, Any]
    # __dict__ is used by the __getitem__ implementation
    __dict__: Dict[str, Any]

    # Methods and properties added by add_shape_dtype_len
    @property
    def shape(self) -> shape_tuple:
        """The shape of the data in the object, as a dynamically-generated namedtuple."""
        ...

    @property
    def dtype(self) -> dtype_tuple:
        """The dtype of the data in the object, as a dynamically-generated namedtuple."""
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
    def batch_shape(self) -> TypingTuple[int, ...]:
        ...

    def reshape(self: T, new_shape: TypingTuple[int, ...]) -> T:
        ...

    def flatten(self: T) -> T:
        ...

    @classmethod
    def random(
        cls: Type[T], shape: TypingTuple[int, ...] = ..., key: chex.PRNGKey = ...
    ) -> T:  # Ellipsis for default value
        ...

    def padding_as_batch(self: T, batch_shape: TypingTuple[int, ...]) -> T:
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

    # Method added by add_comparison_operators
    def __eq__(self, other: Any) -> bool:
        ...

    def __ne__(self, other: Any) -> bool:
        ...


class AtIndexer(Protocol[T]):
    def __getitem__(self: T, index: Any) -> "Updater":
        ...


class Updater(Protocol[T]):
    def set(self: T, value: Any) -> T:
        ...

    def set_as_condition(self: T, condition: chex.Array, value: Any) -> T:
        ...
