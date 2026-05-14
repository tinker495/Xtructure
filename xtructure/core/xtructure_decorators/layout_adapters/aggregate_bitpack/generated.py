"""Generated aggregate bitpack class registration helpers."""

from __future__ import annotations

import sys

GENERATED_CLASS_ATTR = "__xtructure_generated_class__"
GENERATED_ROLE_ATTR = "__xtructure_generated_role__"
GENERATED_PACKED_ROLE = "aggregate_packed"
GENERATED_UNPACKED_VIEW_ROLE = "aggregate_unpacked_view"


def register_generated_class(cls: type, *, role: str) -> type:
    """Mark and register a generated aggregate class in its defining module.

    Xtructure IO stores only ``__module__`` and ``__name__`` metadata. Generated
    aggregate classes therefore need a module-level binding to be loadable. We
    allow generated classes to replace earlier generated versions of themselves
    so module reloads and notebook re-execution remain compatible, but we keep
    rejecting user-defined name collisions.
    """
    setattr(cls, GENERATED_CLASS_ATTR, True)
    setattr(cls, GENERATED_ROLE_ATTR, role)

    module = sys.modules.get(cls.__module__)
    if module is None:
        return cls

    existing = getattr(module, cls.__name__, None)
    if existing is not None and existing is not cls:
        existing_is_same_generated_role = (
            getattr(existing, GENERATED_CLASS_ATTR, False)
            and getattr(existing, GENERATED_ROLE_ATTR, None) == role
        )
        if not existing_is_same_generated_role:
            raise TypeError(
                f"Cannot register generated xtructure class {cls.__module__}.{cls.__name__}: "
                "module already defines a different non-generated object with that name."
            )

    setattr(module, cls.__name__, cls)
    return cls


__all__ = [
    "GENERATED_CLASS_ATTR",
    "GENERATED_PACKED_ROLE",
    "GENERATED_ROLE_ATTR",
    "GENERATED_UNPACKED_VIEW_ROLE",
    "register_generated_class",
]
