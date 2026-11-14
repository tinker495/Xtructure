from collections.abc import Callable
from operator import attrgetter
from typing import Any, Generic, TypeVar

import jax.numpy as jnp

from xtructure.core.layout.types import AdapterFieldPlan

T = TypeVar("T")


def _tuple_getter_values(getter, instance, field_count: int) -> tuple[Any, ...]:
    if field_count == 0:
        return ()
    values = getter(instance)
    if field_count == 1:
        return (values,)
    return values


def _compile_field_setter(
    cls: type,
    field_plans: tuple[AdapterFieldPlan, ...],
    *,
    mode: str,
):
    """Compile a per-field setter for ``cls``.

    ``mode`` selects the contract:
    - ``'set'`` generates ``obj.at[indices].set(value)`` and returns ``_compiled_set``.
    - ``'set_as_condition'`` branches on ``field_kind`` (nested → recursive
      ``set_as_condition``, primitive → ``_UOC``) and returns ``_compiled_condition_set``.
    """
    from ...xtructure_numpy.array_ops import _update_array_on_condition

    if mode == "set":
        fn_name = "_compiled_set"
        signature = "def _compiled_set(obj_instance, indices, values_to_set):"
        value_var = "values_to_set"
    elif mode == "set_as_condition":
        fn_name = "_compiled_condition_set"
        signature = (
            "def _compiled_condition_set("
            "obj_instance, indices, condition, value_to_conditionally_set):"
        )
        value_var = "value_to_conditionally_set"
    else:
        raise ValueError(
            "_compile_field_setter mode must be 'set' or 'set_as_condition', " f"got {mode!r}."
        )

    field_names = tuple(plan.name for plan in field_plans)
    field_count = len(field_names)

    if field_count == 0:
        # Empty dataclass: return a no-arg constructor wrapper directly.
        if mode == "set":

            def _compiled_set(obj_instance, indices, values_to_set):
                return cls()

            return _compiled_set

        def _compiled_condition_set(obj_instance, indices, condition, value_to_conditionally_set):
            return cls()

        return _compiled_condition_set

    field_getter = attrgetter(*field_names)

    def _per_field_assignment(i: int, plan: AdapterFieldPlan) -> str:
        if mode == "set":
            return f"current_{i}.at[indices].set(value_{i})"
        if plan.field_kind == "nested":
            return f"current_{i}.at[indices].set_as_condition(condition, value_{i})"
        return f"_UOC(current_{i}, indices, condition, value_{i})"

    lines = [
        signature,
        f"    is_value_instance = isinstance({value_var}, _CLS)",
        "    if is_value_instance:",
        f"        values_tuple = _TUPLE_GETTER_VALUES("
        f"_FIELD_GETTER, {value_var}, _FIELD_COUNT)",
        "    else:",
        f"        values_tuple = ({value_var},) * _FIELD_COUNT",
        "    new_field_data = {}",
    ]
    for i, plan in enumerate(field_plans):
        lines.append(f"    current_{i} = obj_instance.{plan.name}")
        lines.append(f"    value_{i} = values_tuple[{i}]")
        lines.append(f"    new_field_data[{plan.name!r}] = {_per_field_assignment(i, plan)}")
    lines.append("    return _CLS(**new_field_data)")

    namespace = {
        "_CLS": cls,
        "_FIELD_COUNT": field_count,
        "_FIELD_GETTER": field_getter,
        "_TUPLE_GETTER_VALUES": _tuple_getter_values,
        "_UOC": _update_array_on_condition,
    }
    exec("\n".join(lines), namespace)
    return namespace[fn_name]


class _Updater(Generic[T]):
    __slots__ = ("obj_instance", "indices", "_set", "_set_as_condition")

    def __init__(
        self,
        obj_instance: T,
        index: Any,
        compiled_set: Callable[[T, Any, Any], T],
        compiled_set_as_condition: Callable[[T, Any, Any, Any], T],
    ):
        self.obj_instance = obj_instance
        self.indices = index
        self._set = compiled_set
        self._set_as_condition = compiled_set_as_condition

    def set(self, values_to_set: Any) -> T:
        return self._set(self.obj_instance, self.indices, values_to_set)

    def set_as_condition(self, condition: jnp.ndarray, value_to_conditionally_set: Any) -> T:
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
        return self._set_as_condition(
            self.obj_instance,
            self.indices,
            condition,
            value_to_conditionally_set,
        )


class _AtIndexer(Generic[T]):
    __slots__ = ("obj_instance", "_compiled_set", "_compiled_set_as_condition")

    def __init__(
        self,
        obj_instance: T,
        compiled_set: Callable[[T, Any, Any], T],
        compiled_set_as_condition: Callable[[T, Any, Any, Any], T],
    ):
        self.obj_instance = obj_instance
        self._compiled_set = compiled_set
        self._compiled_set_as_condition = compiled_set_as_condition

    def __getitem__(self, index: Any) -> _Updater[T]:
        return _Updater(
            self.obj_instance, index, self._compiled_set, self._compiled_set_as_condition
        )
