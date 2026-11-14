import builtins
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, Sequence
from typing import Tuple as TypingTuple
from typing import TypeVar

import chex
import jax
from typing_extensions import Self, dataclass_transform

from .layout.types import InstanceLayout
from .structuredtype import StructuredType


# TODO(#43): temporary — accepts decorator-built classes; delete with the decorator.
# Replace the legacy isinstance(Class, Xtructurable) check with issubclass in #43.
class _XtructurableMeta(type):
    def __instancecheck__(cls, instance: Any) -> builtins.bool:
        if cls is Xtructurable:
            return super().__instancecheck__(instance) or bool(
                getattr(instance, "is_xtructed", False)
            )
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass: Any) -> builtins.bool:
        if cls is Xtructurable:
            return super().__subclasscheck__(subclass) or bool(
                getattr(subclass, "is_xtructed", False)
            )
        return super().__subclasscheck__(subclass)


@dataclass_transform()
class Xtructurable(metaclass=_XtructurableMeta):
    """Define a structured JAX dataclass by inheritance, without a decorator.

    Put ``validate`` and ``bitpack`` options in the class header. Applying the
    legacy ``xtructure_dataclass`` decorator to a subclass raises ``TypeError``.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        if cls is Xtructurable:
            raise TypeError(
                "Xtructurable cannot be instantiated directly; subclass Xtructurable instead."
            )
        return super().__new__(cls)

    def __init_subclass__(
        cls,
        *,
        validate: builtins.bool = False,
        bitpack: Literal["auto", "aggregate", "field", "off"] = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if any(
            base is not Xtructurable
            and (Xtructurable in base.__mro__ or getattr(base, "is_xtructed", False))
            for base in cls.__mro__[1:]
        ):
            raise TypeError(
                f"{cls.__name__} cannot inherit from an already xtructured class; "
                "use a plain Python mixin or nested FieldDescriptor composition instead."
            )
        from xtructure.core.xtructure_decorators import install_xtructure_members

        install_xtructure_members(cls, validate=validate, bitpack=bitpack)

    def __iter__(self) -> Iterator[Self]:
        """Iterate over the first batch axis; reject non-batched instances."""
        if self.batch_shape == ():
            raise TypeError(f"Cannot iterate over SINGLE {type(self).__name__} instances.")
        return (self[index] for index in range(len(self)))

    if TYPE_CHECKING:
        is_xtructed: ClassVar[builtins.bool]
        default_shape: ClassVar[Any]
        default_dtype: ClassVar[Any]

        @property
        def shape(self) -> TypingTuple[int, ...]:
            ...

        @property
        def layout(self) -> InstanceLayout:
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
        def default(cls, shape: TypingTuple[int, ...] = ()) -> Self:
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

        def transpose(self, axes: Sequence[int] | None = None) -> Self:
            ...

        def swapaxes(self, axis1: int, axis2: int) -> Self:
            ...

        @classmethod
        def random(cls, shape: TypingTuple[int, ...] = (), key: chex.PRNGKey | None = None) -> Self:
            ...

        def padding_as_batch(self, batch_shape: TypingTuple[int, ...]) -> Self:
            ...

        def __str__(self, **kwargs: Any) -> builtins.str:
            ...

        def str(self, **kwargs: Any) -> builtins.str:
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

        def __eq__(self, other: Any) -> builtins.bool:
            ...

        def __ne__(self, other: Any) -> builtins.bool:
            ...

        def save(self, path: builtins.str, *, packed: builtins.bool = True) -> None:
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


_IndexT = TypeVar("_IndexT")


class AtIndexer(Protocol[_IndexT]):
    def __getitem__(self: _IndexT, index: Any) -> "Updater":
        ...


class Updater(Protocol[_IndexT]):
    def set(self: _IndexT, value: Any) -> _IndexT:
        ...

    def set_as_condition(self: _IndexT, condition: chex.Array, value: Any) -> _IndexT:
        ...
