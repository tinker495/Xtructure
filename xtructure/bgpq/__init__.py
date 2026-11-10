from __future__ import annotations

from typing import TYPE_CHECKING

from xtructure._lazy_imports import lazy_dir, load_lazy_export

if TYPE_CHECKING:
    from .bgpq import BGPQ as BGPQ

_EXPORTS = {"BGPQ": (".bgpq", "BGPQ")}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
