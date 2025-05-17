from typing import Any, Type, TypeVar

import jax.numpy as jnp

T = TypeVar("T")


class _Updater:
    def __init__(self, obj_instance, index):
        self.obj_instance = obj_instance
        self.indices = index
        self.cls = obj_instance.__class__

    def set(self, values_to_set):
        new_field_data = {}

        if not hasattr(self.cls, "__dataclass_fields__"):
            raise TypeError(
                f"Class {self.cls.__name__} is not a recognized dataclass or does not have __dataclass_fields__. "
                f"The .at[...].set(...) feature expects a dataclass structure."
            )

        for field_name in self.cls.__dataclass_fields__:
            current_field_value = getattr(self.obj_instance, field_name)

            try:
                updater_ref = current_field_value.at[self.indices]
                if hasattr(updater_ref, "set"):
                    value_for_this_field = None
                    if isinstance(values_to_set, self.cls):
                        value_for_this_field = getattr(values_to_set, field_name)
                    else:
                        value_for_this_field = values_to_set

                    new_field_data[field_name] = updater_ref.set(value_for_this_field)
                else:
                    new_field_data[field_name] = current_field_value
            except Exception:
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

        for field_name in self.cls.__dataclass_fields__:
            original_field_value = getattr(self.obj_instance, field_name)

            update_val_for_this_field_if_true = None
            if isinstance(value_to_conditionally_set, self.cls):
                update_val_for_this_field_if_true = getattr(value_to_conditionally_set, field_name)
            else:
                update_val_for_this_field_if_true = value_to_conditionally_set

            try:
                if isinstance(getattr(original_field_value, "at", None), AtIndexer):
                    nested_updater = original_field_value.at[self.indices]
                    new_field_data[field_name] = nested_updater.set_as_condition(
                        condition, update_val_for_this_field_if_true
                    )
                elif hasattr(original_field_value, "at") and hasattr(
                    original_field_value.at[self.indices], "set"
                ):
                    original_slice_of_field = original_field_value[self.indices]

                    # Ensure condition is a JAX array to get its ndim property
                    cond_array = jnp.asarray(condition)
                    data_rank = original_slice_of_field.ndim
                    condition_rank = cond_array.ndim

                    reshaped_cond = cond_array
                    if data_rank > condition_rank:
                        num_new_axes = data_rank - condition_rank
                        reshaped_cond = cond_array.reshape(cond_array.shape + (1,) * num_new_axes)
                    # If condition_rank >= data_rank, jnp.where will handle broadcasting or error appropriately.

                    conditionally_updated_slice = jnp.where(
                        reshaped_cond, update_val_for_this_field_if_true, original_slice_of_field
                    )
                    new_field_data[field_name] = original_field_value.at[self.indices].set(
                        conditionally_updated_slice
                    )
                else:
                    new_field_data[field_name] = original_field_value
            except Exception as e:
                import sys

                print(
                    f"Warning: Could not apply conditional set to field '{field_name}' "
                    f"of class '{self.cls.__name__}'. Error: {e}",
                    file=sys.stderr,
                )
                new_field_data[field_name] = original_field_value

        return self.cls(**new_field_data)


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

    def getitem(self, index):
        """Support indexing operations on the dataclass"""
        new_values = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, "__getitem__"):
                new_values[field_name] = field_value[index]
            else:
                new_values[field_name] = field_value
        return cls(**new_values)

    setattr(cls, "__getitem__", getitem)
    setattr(cls, "at", property(AtIndexer))

    return cls
