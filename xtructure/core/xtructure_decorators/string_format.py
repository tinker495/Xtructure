from typing import Type, TypeVar

import jax.numpy as jnp
from tabulate import tabulate

from xtructure.core.structuredtype import StructuredType

from .annotate import MAX_PRINT_BATCH_SIZE, SHOW_BATCH_SIZE

T = TypeVar("T")


def add_string_representation_methods(cls: Type[T]) -> Type[T]:
    """
    Adds custom `__str__` and `str` methods to the class for generating
    a more informative string representation.

    It handles instances categorized by `structured_type` differently:
    - `SINGLE`: Uses the original `__str__` (or `repr` if basic) of the instance.
    - `BATCHED`: Provides a summarized view if the batch is large, showing
      the first few and last few elements, along with the batch shape.
      Uses `tabulate` for formatting.
    - `UNSTRUCTURED`: Indicates that the data is unstructured relative to its
      default shape.
    """

    # Capture the class's __str__ method as it exists *before* this decorator replaces it.
    # This will typically be the __str__ provided by chex.dataclass (similar to its __repr__),
    # or a user-defined __str__ if the user added one before @xtructure_data.
    _original_str_method = getattr(cls, "__str__", None)

    # Determine the function to use for formatting a single item.
    # If the original __str__ is just the basic one from `object`, it's not very informative.
    # In such cases, or if no __str__ was found, `repr` is a better fallback for dataclasses.
    if _original_str_method is None or _original_str_method == object.__str__:
        _single_item_formatter = _custom_pretty_formatter
    else:
        # Use the captured original __str__ method.
        def _single_item_formatter(item, **k):
            return _original_str_method(item, **k)

        # Note: Original __str__ methods typically don't take **kwargs.
        # If kwargs support is needed for the single item formatter,
        # the user would need to define a specific method and the decorator would look for that.
        # For now, we assume the original __str__ doesn't use kwargs from get_str.

    def get_str(self, use_kwargs: bool = False, **kwargs) -> str:
        # This 'self' is an instance of the decorated class 'cls'
        # 'kwargs' are passed from the print(instance) or str(instance) call.

        structured_type = self.structured_type  # This must be a valid property

        if structured_type == StructuredType.SINGLE:
            # For a single item, call the chosen formatter.
            if use_kwargs:
                return _single_item_formatter(self, **kwargs)
            else:
                return _single_item_formatter(self)  # **kwargs will be an empty dict

        elif structured_type == StructuredType.BATCHED:
            batch_shape = self.shape.batch
            batch_len_val = (
                jnp.prod(jnp.array(batch_shape)) if len(batch_shape) != 1 else batch_shape[0]
            )
            py_batch_len = int(batch_len_val)

            results = []
            if py_batch_len <= MAX_PRINT_BATCH_SIZE:
                for i in range(py_batch_len):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state_slice = self[index]
                    # kwargs_idx = {k: v[index] for k, v in kwargs.items()} # Index kwargs if they are batched
                    # For now, assume single_item_formatter doesn't use these indexed kwargs
                    if use_kwargs:
                        results.append(_single_item_formatter(current_state_slice, **kwargs))
                    else:
                        results.append(_single_item_formatter(current_state_slice))
            else:
                for i in range(SHOW_BATCH_SIZE):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state_slice = self[index]
                    if use_kwargs:
                        results.append(_single_item_formatter(current_state_slice, **kwargs))
                    else:
                        results.append(_single_item_formatter(current_state_slice))

                results.append("...\n(batch : " + f"{batch_shape})")

                for i in range(py_batch_len - SHOW_BATCH_SIZE, py_batch_len):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state_slice = self[index]
                    if use_kwargs:
                        results.append(_single_item_formatter(current_state_slice, **kwargs))
                    else:
                        results.append(_single_item_formatter(current_state_slice))
            return tabulate([results], tablefmt="plain")
        else:  # UNSTRUCTURED or any other case
            # Fallback for unstructured or unexpected types to avoid errors,
            # or re-raise the original error if preferred.
            # The original code raised: ValueError(f"State is not structured: {self.shape} != {self.default_shape}")
            # Using repr as a safe fallback:
            return f"<Unstructured {cls.__name__} data, shape: {self.shape}, default_shape: {self.default_shape}>"

    setattr(cls, "__str__", lambda self, **kwargs: get_str(self, use_kwargs=False, **kwargs))
    setattr(
        cls, "str", lambda self, **kwargs: get_str(self, use_kwargs=True, **kwargs)
    )  # Alias .str to the new __str__
    return cls


def _custom_pretty_formatter(item, **_kwargs):  # Accepts and ignores _kwargs for now
    class_name = item.__class__.__name__

    field_values = {}
    # Prioritize __dataclass_fields__ for declared fields in dataclasses
    if hasattr(item, "__dataclass_fields__"):
        for field_name_df in getattr(item, "__dataclass_fields__", {}).keys():
            try:
                field_values[field_name_df] = getattr(item, field_name_df)
            except AttributeError:
                # Field declared but not present; should be rare for dataclasses
                pass
    elif hasattr(item, "__dict__"):
        # Fallback for non-dataclasses or if __dataclass_fields__ is not found/empty
        field_values = item.__dict__
    else:
        # No way to access fields, fallback to simple repr
        return repr(item)

    if not field_values:
        return f"{class_name}()"

    parts = []
    for name, value in field_values.items():
        try:
            value_str = str(value)  # Use str() to leverage our enhanced __str__ for nested items
        except Exception:
            value_str = "<error converting value to string>"

        if "\n" in value_str:
            # Indent all lines of the multi-line value string for better readability
            indented_value = "\n".join(["    " + line for line in value_str.split("\n")])
            parts.append(f"  {name}: \n{indented_value}")
        else:
            parts.append(f"  {name}: {value_str}")

    return f"{class_name}(\n" + ",\n".join(parts) + "\n)"
