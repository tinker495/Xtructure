from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from .protocol import Xtructurable


def is_xtructure_dataclass_type(value: Any) -> TypeIs[type[Xtructurable]]:
    """Return True if `value` is an xtructure dataclass type.

    Accepts `Xtructurable` subclasses and classes still using the transitional
    `xtructure_dataclass` decorator, which marks both with `is_xtructed = True`.
    """
    return isinstance(value, type) and bool(getattr(value, "is_xtructed", False))


def is_xtructure_dataclass_instance(value: Any) -> TypeIs[Xtructurable]:
    """Return True if `value` is an instance of an xtructure dataclass type.

    Unlike ``isinstance(value, Xtructurable)``, this also accepts instances of
    classes still using the transitional `xtructure_dataclass` decorator.
    """
    return bool(getattr(value, "is_xtructed", False))
