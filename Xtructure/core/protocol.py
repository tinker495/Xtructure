from typing import Any, Dict, Type, Protocol, Tuple as TypingTuple, TypeVar

from .structuredtype import StructuredType

T = TypeVar("T")

# Protocol defining the interface added by @xtructure_data
class Xtructurable(Protocol[T]):
    # Fields from the original class that chex.dataclass would process
    # These are implicitly part of T. For the protocol to be complete,
    # it assumes T will have __annotations__.
    __annotations__: Dict[str, Any]
    # __dict__ is used by the __getitem__ implementation
    __dict__: Dict[str, Any]


    # Methods and properties added by add_shape_dtype_getitem_len
    @property
    def shape(self) -> Any: # Actual type is a dynamically generated namedtuple
        ...
    @property
    def dtype(self) -> Any: # Actual type is a dynamically generated namedtuple
        ...
    def __getitem__(self: T, index: Any) -> T:
        ...
    def __len__(self) -> int:
        ...

    # Methods and properties added by add_structure_utilities_and_random
    # Assumes the class T has a 'default' classmethod as per the decorator's assertion
    @classmethod
    def default(cls: Type[T], shape: Any = ...) -> T:
        ...
    @property
    def default_shape(self) -> Any: # Derived from self.default().shape
        ...
    @property
    def structured_type(self) -> 'StructuredType': # Forward reference for StructuredType
        ...
    @property
    def batch_shape(self) -> TypingTuple[int, ...]:
        ...
    def reshape(self: T, new_shape: TypingTuple[int, ...]) -> T:
        ...
    def flatten(self: T) -> T:
        ...
    @classmethod
    def random(cls: Type[T], shape: TypingTuple[int, ...] = ..., key: Any = ...) -> T: # Ellipsis for default value
        ...

    # Methods and properties added by add_string_representation_methods
    def __str__(self) -> str: # The actual implementation takes **kwargs, but signature can be simpler for Protocol
        ...
    def str(self) -> str: # Alias for __str__
        ...
