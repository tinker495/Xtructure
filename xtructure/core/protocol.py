import builtins
import dataclasses
import inspect
import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterator,
    Literal,
    Protocol,
    Self,
    Sequence,
)
from typing import Tuple as TypingTuple
from typing import TypeVar, get_origin

import chex
import jax
from typing_extensions import dataclass_transform

from .structuredtype import StructuredType

T_co = TypeVar("T_co", covariant=True)


# Abstract views (``abstract=True``): annotation-only bases of runtime-built classes.
_ABSTRACT_VIEWS: "weakref.WeakSet[type]" = weakref.WeakSet()


def _reject_view_instantiation(self: object, *args: Any, **kwargs: Any) -> None:
    raise TypeError(
        f"{type(self).__name__} is an abstract view (abstract=True) and cannot be "
        "instantiated; instantiate a concrete subclass"
    )


def _require_view_fields(cls: type) -> None:
    """Fail at class creation when ``cls`` leaves a field of an abstract view base undeclared."""
    declared = {field.name for field in dataclasses.fields(cls)}
    for base in cls.__mro__[1:]:
        if base not in _ABSTRACT_VIEWS:
            continue
        missing = sorted(
            name
            for name, annotation in inspect.get_annotations(base, eval_str=True).items()
            if annotation is not ClassVar
            and get_origin(annotation) is not ClassVar
            and name not in declared
        )
        if missing:
            raise TypeError(
                f"{cls.__name__} subclasses the abstract view {base.__name__} but does not "
                f"declare its field(s) {', '.join(missing)}"
            )


@dataclass_transform()
class Xtructurable:
    """Base class of xtructure dataclasses::

        class Node(Xtructurable, bitpack="off"):
            cost: Annotated[jax.Array, FieldDescriptor.scalar(dtype=jnp.float32)]

    Subclassing applies the xtructure dataclass transformation (``validate`` and
    ``bitpack`` are class keywords) and, because the injected members are declared
    here, exposes them to type checkers, which ignore class-decorator return types.
    The declarations exist only under ``TYPE_CHECKING``; at runtime the
    transformation installs the real members. ``isinstance(x, Xtructurable)`` tells
    whether a value is an xtructure dataclass.

    ``abstract=True`` declares a typed view that is not transformed: its plain
    field annotations describe, for type checkers, the classes built from it at
    runtime (for example per-instance classes whose shapes are only known then)::

        class BoardState(Xtructurable, abstract=True):
            board: jax.Array

        def make_state_class(n: int) -> type[BoardState]:
            class State(BoardState):
                board: Annotated[jax.Array, FieldDescriptor.tensor(dtype=jnp.uint8, shape=(n,))]

            return State

    A view exists only for annotations: instantiating it raises ``TypeError``, it
    takes no ``validate``/``bitpack`` options, and every concrete subclass must
    redeclare all of its fields with a ``FieldDescriptor``.
    """

    def __init_subclass__(
        cls,
        *,
        abstract: bool = False,
        validate: bool | None = None,
        bitpack: Literal["auto", "aggregate", "field", "off"] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if abstract:
            if validate is not None or bitpack is not None:
                raise TypeError(
                    f"{cls.__name__} is an abstract view; validate/bitpack apply to the "
                    "concrete subclasses that redeclare its fields"
                )
            setattr(cls, "__init__", _reject_view_instantiation)
            _ABSTRACT_VIEWS.add(cls)
            return
        from .xtructure_decorators import install_xtructure_members

        install_xtructure_members(
            cls,
            validate=False if validate is None else validate,
            bitpack="auto" if bitpack is None else bitpack,
        )
        _require_view_fields(cls)

    if TYPE_CHECKING:
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

        def __iter__(self) -> Iterator[Self]:
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

        def str(self, **kwargs: Any) -> builtins.str:
            ...

        def __str__(self, **kwargs: Any) -> builtins.str:
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

        # Like jax.Array: comparisons return a (reduced, scalar) array, not a bool.
        def __eq__(self, other: Any) -> jax.Array:  # pyrefly: ignore[bad-override]
            ...

        def __ne__(self, other: Any) -> jax.Array:  # pyrefly: ignore[bad-override]
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


# Element type of the xtructure containers and other APIs generic over a dataclass.
XtructurableT = TypeVar("XtructurableT", bound=Xtructurable)


class AtIndexer(Protocol[T_co]):
    def __getitem__(self, index: Any) -> "Updater[T_co]":
        ...


class Updater(Protocol[T_co]):
    def set(self, value: Any) -> T_co:
        ...

    def set_as_condition(self, condition: chex.Array, value: Any) -> T_co:
        ...
