from typing import Type, TypeVar

from xtructure.core.display import (
    MAX_PRINT_BATCH_SIZE,
    SHOW_BATCH_SIZE,
    BatchedRenderer,
)
from xtructure.core.structuredtype import StructuredType

T = TypeVar("T")


def add_string_representation_methods(cls: Type[T]) -> Type[T]:
    """Attach ``__str__`` / ``str`` that dispatch on ``structured_type``.

    - ``SINGLE``: forwards to the user's ``__str__`` (or
      :func:`_custom_pretty_formatter` when none was defined).
    - ``BATCHED``: delegates to a class-bound :class:`BatchedRenderer`.
    - ``UNSTRUCTURED``: inline one-liner reporting the shape mismatch.

    ``__str__`` and ``str`` are the same callable; kwargs passed to ``str``
    flow through to the single-item formatter (SINGLE) or to
    :meth:`BatchedRenderer.render` after ``max_size`` / ``show_size``
    are extracted.
    """
    original_str = getattr(cls, "__str__", None)
    if original_str is None or original_str is object.__str__:
        single_formatter = _custom_pretty_formatter
    else:

        def single_formatter(item, **kwargs):
            return original_str(item, **kwargs)

    renderer = BatchedRenderer(single_formatter)

    def __str__(self, **kwargs) -> str:
        structured_type = self.structured_type

        if structured_type == StructuredType.SINGLE:
            return single_formatter(self, **kwargs)

        if structured_type == StructuredType.BATCHED:
            max_size = kwargs.pop("max_size", MAX_PRINT_BATCH_SIZE)
            show_size = kwargs.pop("show_size", SHOW_BATCH_SIZE)
            return renderer.render(
                self,
                max_size=max_size,
                show_size=show_size,
                title=f"Batched {cls.__name__}",
                **kwargs,
            )

        return (
            f"<Unstructured {cls.__name__} data, shape: {self.shape}, "
            f"default_shape: {self.default_shape}>"
        )

    setattr(cls, "__str__", __str__)
    setattr(cls, "str", __str__)
    return cls


def _custom_pretty_formatter(item, **_kwargs) -> str:
    class_name = item.__class__.__name__

    field_values = {}
    if hasattr(item, "__dataclass_fields__"):
        for field_name_df in getattr(item, "__dataclass_fields__", {}).keys():
            try:
                field_values[field_name_df] = getattr(item, field_name_df)
            except AttributeError:
                pass
    elif hasattr(item, "__dict__"):
        field_values = item.__dict__
    else:
        return repr(item)

    if not field_values:
        return f"{class_name}()"

    parts = []
    for name, value in field_values.items():
        try:
            value_str = str(value)
        except Exception:
            value_str = "<error converting value to string>"

        if "\n" in value_str:
            indented_value = "\n".join(["    " + line for line in value_str.split("\n")])
            parts.append(f"  {name}: \n{indented_value}")
        else:
            parts.append(f"  {name}: {value_str}")

    return f"{class_name}(\n" + ",\n".join(parts) + "\n)"
