import builtins
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, Self, Sequence
from typing import Tuple as TypingTuple
from typing import TypeVar

import chex
import jax
from typing_extensions import dataclass_transform

from .structuredtype import StructuredType

T_co = TypeVar("T_co", covariant=True)


@dataclass_transform()
class Xtructurable:
    """Base class of xtructure dataclasses::

        class Node(Xtructurable, bitpack="off"):
            cost: Annotated[jax.Array, FieldDescriptor.scalar(dtype=jnp.float32)]

    Subclassing applies the xtructure dataclass transformation (``validate`` and
    ``bitpack`` are class keywords) and, because the injected members are declared
    here, exposes them to type checkers, which ignore class-decorator return types.
    The declarations exist only under ``TYPE_CHECKING``; at runtime the
    transformation installs the real members. ``isinstance(x, Xtructurable)`` is an
    ordinary subclass check; ``is_xtructure_dataclass_instance`` also accepts
    classes still using the transitional ``xtructure_dataclass`` decorator.
    """

    def __init_subclass__(
        cls,
        *,
        validate: bool = False,
        bitpack: Literal["auto", "aggregate", "field", "off"] = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        from .xtructure_decorators import install_xtructure_members

        installed = install_xtructure_members(cls, validate=validate, bitpack=bitpack)
        assert installed is cls, "the xtructure transformation must work in place"

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

        def transpose(self, axes: Sequence[int] | None = ...) -> Self:
            ...

        def swapaxes(self, axis1: int, axis2: int) -> Self:
            ...

        @classmethod
        def random(cls, shape: TypingTuple[int, ...] = ..., key: chex.PRNGKey = ...) -> Self:
            ...

        def padding_as_batch(self, batch_shape: TypingTuple[int, ...]) -> Self:
            ...

        def str(self) -> builtins.str:
            ...

        @property
        def at(self) -> "AtIndexer[Self]":
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


class AtIndexer(Protocol[T_co]):
    def __getitem__(self, index: Any) -> "Updater[T_co]":
        ...


class Updater(Protocol[T_co]):
    def set(self, value: Any) -> T_co:
        ...

    def set_as_condition(self, condition: chex.Array, value: Any) -> T_co:
        ...
