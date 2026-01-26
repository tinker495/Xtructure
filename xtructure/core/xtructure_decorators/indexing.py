from operator import attrgetter
from typing import Any, Type, TypeVar

import jax.numpy as jnp

from ..xtructure_numpy.array_ops import _update_array_on_condition

T = TypeVar("T")


class _Updater:
    def __init__(self, obj_instance, index):
        self.obj_instance = obj_instance
        self.indices = index
        self.cls = obj_instance.__class__
        # Prefer per-class cached field order/getter injected by add_indexing_methods.
        cached_names = getattr(self.cls, "__xtructure_field_names__", None)
        cached_getter = getattr(self.cls, "__xtructure_field_getter__", None)
        if cached_names is not None and cached_getter is not None:
            self._field_names = cached_names
            self._field_getter = cached_getter
        else:
            # Fallback: compute from dataclass/annotations.
            if hasattr(self.cls, "__dataclass_fields__"):
                self._field_names = tuple(self.cls.__dataclass_fields__.keys())
            else:
                self._field_names = tuple(getattr(self.cls, "__annotations__", {}).keys())
            self._field_getter = attrgetter(*self._field_names) if self._field_names else None

    def set(self, values_to_set):
        new_field_data = {}

        if not hasattr(self.cls, "__dataclass_fields__"):
            raise TypeError(
                f"Class {self.cls.__name__} is not a recognized dataclass or does not have __dataclass_fields__. "
                f"The .at[...].set(...) feature expects a dataclass structure."
            )

        is_value_instance = isinstance(values_to_set, self.cls)
        values_getter = self._field_getter if is_value_instance else None
        values_tuple = None
        if is_value_instance and values_getter is not None:
            values_tuple = values_getter(values_to_set)
            if len(self._field_names) == 1:
                values_tuple = (values_tuple,)

        instance_values = self._field_getter(self.obj_instance) if self._field_getter else ()
        if len(self._field_names) == 1:
            instance_values = (instance_values,)

        for i, (field_name, current_field_value) in enumerate(
            zip(self._field_names, instance_values)
        ):
            try:
                # Most common fast path: arrays and xtructure instances both expose `.at[...]`.
                updater_ref = current_field_value.at[self.indices]
                if hasattr(updater_ref, "set"):
                    value_for_this_field = (
                        values_tuple[i] if values_tuple is not None else values_to_set
                    )
                    new_field_data[field_name] = updater_ref.set(value_for_this_field)
                else:
                    new_field_data[field_name] = current_field_value
            except (AttributeError, TypeError, IndexError, KeyError, ValueError):
                # Preserve legacy behavior: if a field can't be updated, keep original value.
                new_field_data[field_name] = current_field_value

        return self.cls(**new_field_data)

    def set_as_condition(self, condition: jnp.ndarray, value_to_conditionally_set: Any):
        """
        Sets parts of the fields of the dataclass instance based on a condition.
        This is an out-of-place update.

        Args:
            condition: A JAX boolean array. Its shape should be compatible with
                       the slice of the fields selected by `self.indices` through broadcasting.
                       It determines element-wise whether to use the new value
                       or the original value.
            value_to_conditionally_set: The value(s) to set if the condition is true.
                                       - If it's an instance of the same dataclass type (`self.cls`),
                                         the corresponding fields from this instance are used for updates.
                                       - Otherwise (e.g., a scalar or a JAX array), this value is used
                                         for updating all applicable fields (it must be broadcast-compatible
                                         with the slice of each field).
        Returns:
            A new instance of the dataclass with updated fields.
        """
        new_field_data = {}

        if not hasattr(self.cls, "__dataclass_fields__"):
            raise TypeError(
                f"Class {self.cls.__name__} is not a recognized dataclass or does not have __dataclass_fields__. "
                f"The .at[...].set_as_condition(...) feature expects a dataclass structure."
            )

        is_value_instance = isinstance(value_to_conditionally_set, self.cls)
        values_getter = self._field_getter if is_value_instance else None
        values_tuple = None
        if is_value_instance and values_getter is not None:
            values_tuple = values_getter(value_to_conditionally_set)
            if len(self._field_names) == 1:
                values_tuple = (values_tuple,)

        instance_values = self._field_getter(self.obj_instance) if self._field_getter else ()
        if len(self._field_names) == 1:
            instance_values = (instance_values,)

        for i, (field_name, original_field_value) in enumerate(
            zip(self._field_names, instance_values)
        ):
            update_val_for_this_field_if_true = (
                values_tuple[i] if values_tuple is not None else value_to_conditionally_set
            )

            try:
                if hasattr(original_field_value, "at"):
                    nested_updater = original_field_value.at[self.indices]

                    # Recursive dataclass update path.
                    if hasattr(nested_updater, "set_as_condition"):
                        new_field_data[field_name] = nested_updater.set_as_condition(
                            condition, update_val_for_this_field_if_true
                        )
                        continue

                    # Array-like update path.
                    if hasattr(nested_updater, "set"):
                        new_field_data[field_name] = _update_array_on_condition(
                            original_field_value,
                            self.indices,
                            condition,
                            update_val_for_this_field_if_true,
                        )
                        continue

                new_field_data[field_name] = original_field_value
            except Exception:
                # Preserve legacy behavior: if a field can't be updated, keep original value.
                new_field_data[field_name] = original_field_value

        return self.cls(**new_field_data)

    def _legacy_set_as_condition(self, condition, value_to_conditionally_set):
        # Kept for compatibility if needed, but redundant with the above revert
        return self.set_as_condition(condition, value_to_conditionally_set)


class AtIndexer:
    def __init__(self, obj_instance):
        self.obj_instance = obj_instance

    def __getitem__(self, index):
        return _Updater(self.obj_instance, index)


def add_indexing_methods(cls: Type[T]) -> Type[T]:
    """
    Augments the class with an `__getitem__` method for indexing/slicing
    and an `at` property that enables JAX-like out-of-place updates
    (e.g., `instance.at[index].set(value)`).

    The `__getitem__` method allows instances to be indexed, applying the
    index to each field.
    The `at` property provides access to an updater object for specific indices.
    """
    # Pre-compute field order once to avoid per-call introspection.
    if hasattr(cls, "__dataclass_fields__"):
        _field_names = tuple(cls.__dataclass_fields__.keys())
    else:
        _field_names = tuple(getattr(cls, "__annotations__", {}).keys())

    _field_getter = attrgetter(*_field_names) if _field_names else None

    # Expose cached metadata for _Updater (avoids repeated introspection).
    setattr(cls, "__xtructure_field_names__", _field_names)
    setattr(cls, "__xtructure_field_getter__", _field_getter)

    def getitem(self, index):
        """Support indexing operations on the dataclass"""
        if not _field_names:
            return cls()

        values = _field_getter(self)
        if len(_field_names) == 1:
            values = (values,)

        new_values = {}
        for field_name, field_value in zip(_field_names, values):
            if hasattr(field_value, "__getitem__"):
                new_values[field_name] = field_value[index]
            else:
                new_values[field_name] = field_value
        return cls(**new_values)

    setattr(cls, "__getitem__", getitem)
    setattr(cls, "at", property(AtIndexer))

    return cls
