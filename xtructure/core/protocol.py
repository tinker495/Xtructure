import builtins
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self
from typing import Tuple as TypingTuple
from typing import TypeVar

import chex
import jax

from .structuredtype import StructuredType

T = TypeVar("T")


class _XtructurableMeta(type):
    # isinstance/issubclass against Xtructurable itself accept every decorated class,
    # including ones that do not inherit it; checks against subclasses stay nominal.
    def __instancecheck__(cls, instance) -> bool:
        if cls is Xtructurable:
            return bool(getattr(instance, "is_xtructed", False))
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass) -> bool:
        if cls is Xtructurable:
            return bool(getattr(subclass, "is_xtructed", False))
        return super().__subclasscheck__(subclass)


class Xtructurable(metaclass=_XtructurableMeta):
    """Members `xtructure_dataclass` injects; inherit it in decorated classes::

        @xtructure_dataclass
        class Node(Xtructurable):
            cost: Annotated[jax.Array, FieldDescriptor.scalar(dtype=jnp.float32)]

    Type checkers keep a decorated class's own type and ignore a class decorator's
    return annotation, so the injected members are invisible unless they sit in the
    class's MRO. The declarations exist only under ``TYPE_CHECKING``; at runtime this
    is an empty class and the decorator installs the real members. ``isinstance(x,
    Xtructurable)`` is true for every decorated instance, inheriting or not.
    """

    if TYPE_CHECKING:
        is_xtructed: ClassVar[bool]
        default_dtype: ClassVar[Any]

        @property
        def shape(self) -> Any:
            ...

        @property
        def dtype(self) -> Any:
            ...

        @property
        def ndim(self) -> int:
            ...

        def __getitem__(self, index: Any) -> Self:
            ...

        def __len__(self) -> int:
            ...

        @classmethod
        def default(cls, shape: Any = ...) -> Self:
            ...

        @property
        def default_shape(self) -> Any:
            ...

        @property
        def structured_type(self) -> StructuredType:
            ...

        @property
        def batch_shape(self) -> TypingTuple[int, ...] | int:
            ...

        def reshape(self, *new_shape: int | TypingTuple[int, ...]) -> Self:
            ...

        def flatten(self) -> Self:
            ...

        def transpose(self, axes: TypingTuple[int, ...] | None = ...) -> Self:
            ...

        @classmethod
        def random(cls, shape: TypingTuple[int, ...] = ..., key: chex.PRNGKey = ...) -> Self:
            ...

        def padding_as_batch(self, batch_shape: TypingTuple[int, ...]) -> Self:
            ...

        def str(self) -> builtins.str:
            ...

        @property
        def at(self) -> "AtIndexer":
            ...

        @property
        def bytes(self) -> chex.Array:
            ...

        @property
        def uint32ed(self) -> jax.Array:
            ...

        def hash(self, seed: int = 0) -> jax.Array:
            ...

        def hash_with_uint32ed(self, seed: int = 0) -> TypingTuple[jax.Array, jax.Array]:
            ...

        def __eq__(self, other: Any) -> bool:
            ...

        def __ne__(self, other: Any) -> bool:
            ...

        def save(self, path: builtins.str) -> None:
            ...

        @classmethod
        def load(cls, path: builtins.str) -> Self:
            ...

        def check_invariants(self) -> None:
            ...

        @classmethod
        def from_tuple(cls, args: TypingTuple[Any, ...]) -> Self:
            ...

        def to_tuple(self) -> TypingTuple[Any, ...]:
            ...

        def replace(self, **kwargs: Any) -> Self:
            ...


class AtIndexer(Protocol[T]):
    def __getitem__(self: T, index: Any) -> "Updater":
        ...


class Updater(Protocol[T]):
    def set(self: T, value: Any) -> T:
        ...

    def set_as_condition(self: T, condition: chex.Array, value: Any) -> T:
        ...
