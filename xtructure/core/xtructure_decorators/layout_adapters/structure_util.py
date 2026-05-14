from typing import Type, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from xtructure.core.layout import get_type_layout
from xtructure.core.structuredtype import StructuredType

T = TypeVar("T")


def add_structure_utilities(cls: Type[T]) -> Type[T]:
    """
    Augments the class with utility methods and properties related to its
    structural representation (based on a 'default' instance), batch operations,
    and random instance generation.

    Requires the class to have a `default` classmethod, which is used to
    determine default shapes, dtypes, and behaviors.

    Adds:
        - Properties:
            - `default_shape`: Shape of the instance returned by `cls.default()`.
            - `structured_type`: An enum (`StructuredType`) indicating if the
              instance is SINGLE, BATCHED, or UNSTRUCTURED relative to its
              default shape.
            - `batch_shape`: The shape of the batch dimensions if `structured_type`
              is BATCHED.
        - Instance Methods:
            - `reshape(*new_shape)`: Reshapes the batch dimensions of a BATCHED instance.
            - `flatten()`: Flattens the batch dimensions of a BATCHED instance.
            - `transpose(axes=None)`: Transposes only the batch dimensions.
        - Classmethod:
            - `random(shape=(), key=None)`: Generates an instance with random data.
              The `shape` argument specifies the desired batch shape, which is
              prepended to the default field shapes.
    """
    assert hasattr(cls, "default"), "There is no default method."

    type_layout = get_type_layout(cls)
    field_plans = type_layout.adapter_field_plans

    def reshape(self, *new_shape: int | tuple[int, ...]) -> T:
        if len(new_shape) == 0:
            raise ValueError("new_shape must be provided")
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        else:
            new_shape = tuple(new_shape)

        if self.structured_type == StructuredType.BATCHED:
            total_length = np.prod(self.shape.batch)

            # Handle -1 in new_shape by calculating the missing dimension
            new_shape_list = list(new_shape)
            if -1 in new_shape_list:
                # Count how many -1s are in the shape
                minus_one_count = new_shape_list.count(-1)
                if minus_one_count > 1:
                    raise ValueError("Only one -1 is allowed in new_shape")

                # Calculate the product of all non-negative values in new_shape
                non_negative_product = 1
                for dim in new_shape_list:
                    if dim != -1:
                        non_negative_product *= dim

                # Calculate what the -1 should be
                if non_negative_product == 0:
                    raise ValueError("Cannot infer -1 dimension when other dimensions are 0")

                inferred_dim = total_length // non_negative_product
                if total_length % non_negative_product != 0:
                    raise ValueError(
                        f"Total length {total_length} is not divisible by the product of "
                        f"other dimensions {non_negative_product}"
                    )

                # Replace -1 with the calculated dimension
                minus_one_index = new_shape_list.index(-1)
                new_shape_list[minus_one_index] = inferred_dim
                new_shape = tuple(new_shape_list)

            new_total_length = np.prod(new_shape)
            batch_dim = len(self.shape.batch)
            if total_length != new_total_length:
                raise ValueError(
                    f"Total length of the state and new shape does not match: {total_length} != {new_total_length}"
                )
            return jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, new_shape + x.shape[batch_dim:]), self
            )
        else:
            raise ValueError(
                f"Reshape is only supported for BATCHED structured_type. Current type: '{self.structured_type}'."
                f"Shape: {self.shape}, Default Shape: {self.default_shape}"
            )

    def flatten(self):
        if self.structured_type != StructuredType.BATCHED:
            raise ValueError(
                f"Flatten operation is only supported for BATCHED structured types. "
                f"Current type: {self.structured_type}"
            )

        current_batch_shape = self.shape.batch
        # np.prod of an empty tuple array is 1, which is correct for total_length
        # if current_batch_shape is ().
        total_length = np.prod(np.array(current_batch_shape))
        len_current_batch_shape = len(current_batch_shape)

        return jax.tree_util.tree_map(
            # Reshape each leaf: flatten batch dims, keep core dims.
            # core_dims are obtained by stripping batch_dims from the start of x.shape.
            lambda x: jnp.reshape(x, (total_length,) + x.shape[len_current_batch_shape:]),
            self,
        )

    def transpose(self, axes: tuple[int, ...] | None = None) -> T:
        if self.structured_type == StructuredType.UNSTRUCTURED:
            raise ValueError(
                "Transpose operation is only supported for SINGLE or BATCHED structured types. "
                f"Current type: {self.structured_type}"
            )

        batch_shape = self.shape.batch
        batch_ndim = len(batch_shape)

        if axes is None:
            axes = tuple(range(batch_ndim - 1, -1, -1))

        def transpose_batch_only(field):
            field_ndim = field.ndim
            if field_ndim <= batch_ndim:
                return jnp.transpose(field, axes=axes)
            full_axes = list(axes) + list(range(batch_ndim, field_ndim))
            return jnp.transpose(field, axes=full_axes)

        return jax.tree_util.tree_map(transpose_batch_only, self)

    def random(cls, shape=(), key=None):
        if key is None:
            key = jax.random.PRNGKey(0)

        data = {}
        keys = jax.random.split(key, len(field_plans))

        for i, field_plan in enumerate(field_plans):
            field_key = keys[i]
            field_name = field_plan.name
            target_shape = shape + field_plan.intrinsic_shape

            if field_plan.random_kind == "nested":
                nested_class = field_plan.nested_type
                # Recursively call random for the nested xtructure_data class.
                data[field_name] = nested_class.random(shape=target_shape, key=field_key)
            elif field_plan.random_kind == "bits_int":
                generated_bits = jax.random.bits(
                    field_key, shape=target_shape, dtype=field_plan.random_bits_dtype
                )
                if field_plan.random_view_as_signed:
                    data[field_name] = generated_bits.view(field_plan.declared_dtype)
                else:
                    data[field_name] = generated_bits
            elif field_plan.random_kind == "float":
                data[field_name] = jax.random.uniform(
                    field_key, target_shape, dtype=field_plan.random_gen_dtype
                )
            elif field_plan.random_kind == "bool":
                data[field_name] = jax.random.bernoulli(field_key, shape=target_shape)
            else:
                try:
                    data[field_name] = jnp.zeros(target_shape, dtype=field_plan.random_gen_dtype)
                except TypeError:
                    raise NotImplementedError(
                        f"Random generation for dtype {field_plan.random_gen_dtype} "
                        f"(field: {field_name}) is not implemented robustly."
                    )
        return cls(**data)

    def padding_as_batch(self, batch_shape: tuple[int, ...]):
        if self.structured_type != StructuredType.BATCHED or len(self.shape.batch) > 1:
            raise ValueError(
                "Padding as batch operation is only supported for BATCHED structured types "
                "with at most 1 batch dimension. "
                f"Current type: {self.structured_type}, "
                f"Current batch shape: {self.shape.batch}"
            )
        if self.shape.batch == batch_shape:
            return self

        new_default_state = self.default(batch_shape)
        new_default_state = new_default_state.at[: self.shape.batch[0]].set(self)
        return new_default_state

    # add method based on default state
    setattr(cls, "reshape", reshape)
    setattr(cls, "flatten", flatten)
    setattr(cls, "transpose", transpose)
    setattr(cls, "random", classmethod(random))
    setattr(cls, "padding_as_batch", padding_as_batch)
    return cls
