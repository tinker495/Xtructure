from typing import Any, Type, TypeVar

import jax
import jax.numpy as jnp

from xtructure.core.field_descriptors import FieldDescriptor
from xtructure.core.structuredtype import StructuredType

T = TypeVar("T")


def is_nested_xtructure(dtype: Any) -> bool:
    if isinstance(dtype, type):
        if hasattr(dtype, "is_xtructed"):
            return True
        return False
    else:
        return False


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
            - `reshape(new_shape)`: Reshapes the batch dimensions of a BATCHED instance.
            - `flatten()`: Flattens the batch dimensions of a BATCHED instance.
        - Classmethod:
            - `random(shape=(), key=None)`: Generates an instance with random data.
              The `shape` argument specifies the desired batch shape, which is
              prepended to the default field shapes.
    """
    assert hasattr(cls, "default"), "There is no default method."

    field_descriptors: dict[str, FieldDescriptor] = cls.__annotations__
    default_shape = dict([(fn, fd.intrinsic_shape) for fn, fd in field_descriptors.items()])
    default_dtype = dict([(fn, fd.dtype) for fn, fd in field_descriptors.items()])

    # Pre-calculate generation configurations for the random method
    _field_generation_configs = []
    # Ensure consistent order for key splitting, matching __annotations__
    _field_names_for_random = list(cls.__annotations__.keys())

    for field_name_cfg in _field_names_for_random:
        cfg = {}
        cfg["name"] = field_name_cfg
        # Retrieve the dtype or nested dtype tuple for the current field
        actual_dtype_or_nested_dtype_tuple = default_dtype[field_name_cfg]
        cfg["default_field_shape"] = default_shape[field_name_cfg]

        if is_nested_xtructure(actual_dtype_or_nested_dtype_tuple):
            # This field is a nested xtructure_data instance
            cfg["type"] = "xtructure"
            # Store the actual nested class type (e.g., Parent, Current)
            cfg["nested_class_type"] = cls.__annotations__[field_name_cfg].dtype
            # Store the namedtuple of dtypes for the nested structure
            cfg["actual_dtype"] = actual_dtype_or_nested_dtype_tuple
        else:
            # This field is a regular JAX array
            actual_dtype = actual_dtype_or_nested_dtype_tuple  # It's a single JAX dtype here
            cfg["actual_dtype"] = actual_dtype  # Store the single JAX dtype

            if jnp.issubdtype(actual_dtype, jnp.integer):
                cfg["type"] = "bits_int"  # Unified type for all full-range integers via bits
                if jnp.issubdtype(actual_dtype, jnp.unsignedinteger):
                    cfg["bits_gen_dtype"] = actual_dtype  # Generate bits of this same unsigned type
                    cfg["view_as_signed"] = False
                else:  # It's a signed integer
                    unsigned_equivalent_str = f"uint{actual_dtype.itemsize * 8}"
                    cfg["bits_gen_dtype"] = jnp.dtype(
                        unsigned_equivalent_str
                    )  # Generate bits of corresponding unsigned type
                    cfg["view_as_signed"] = True  # And then view them as the actual signed type
            elif jnp.issubdtype(actual_dtype, jnp.floating):
                cfg["type"] = "float"
                cfg["gen_dtype"] = actual_dtype
            elif actual_dtype == jnp.bool_:
                cfg["type"] = "bool"
            else:
                cfg["type"] = "other"  # Fallback
                cfg["gen_dtype"] = actual_dtype
        _field_generation_configs.append(cfg)

    def reshape(self, new_shape: tuple[int, ...]) -> T:
        if self.structured_type == StructuredType.BATCHED:
            total_length = jnp.prod(jnp.array(self.shape.batch))
            new_total_length = jnp.prod(jnp.array(new_shape))
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
        # jnp.prod of an empty tuple array is 1, which is correct for total_length
        # if current_batch_shape is ().
        total_length = jnp.prod(jnp.array(current_batch_shape))
        len_current_batch_shape = len(current_batch_shape)

        return jax.tree_util.tree_map(
            # Reshape each leaf: flatten batch dims, keep core dims.
            # core_dims are obtained by stripping batch_dims from the start of x.shape.
            lambda x: jnp.reshape(x, (total_length,) + x.shape[len_current_batch_shape:]),
            self,
        )

    def random(cls, shape=(), key=None):
        if key is None:
            key = jax.random.PRNGKey(0)

        data = {}
        keys = jax.random.split(key, len(_field_generation_configs))

        for i, cfg in enumerate(_field_generation_configs):
            field_key = keys[i]
            field_name = cfg["name"]

            if cfg["type"] == "xtructure":
                nested_class = cfg["nested_class_type"]
                # For nested xtructures, combine batch shape with field shape
                current_default_shape = cfg["default_field_shape"]
                target_shape = shape + current_default_shape
                # Recursively call random for the nested xtructure_data class.
                data[field_name] = nested_class.random(shape=target_shape, key=field_key)
            else:
                # This branch handles primitive JAX array fields.
                current_default_shape = cfg["default_field_shape"]
                if not isinstance(current_default_shape, tuple):
                    current_default_shape = (
                        current_default_shape,
                    )  # Ensure it's a tuple for concatenation

                target_shape = shape + current_default_shape

                if cfg["type"] == "bits_int":
                    generated_bits = jax.random.bits(
                        field_key, shape=target_shape, dtype=cfg["bits_gen_dtype"]
                    )
                    if cfg["view_as_signed"]:
                        data[field_name] = generated_bits.view(cfg["actual_dtype"])
                    else:
                        data[field_name] = generated_bits
                elif cfg["type"] == "float":
                    data[field_name] = jax.random.uniform(
                        field_key, target_shape, dtype=cfg["gen_dtype"]
                    )
                elif cfg["type"] == "bool":
                    data[field_name] = jax.random.bernoulli(
                        field_key, shape=target_shape  # p=0.5 by default
                    )
                else:  # Fallback for 'other' dtypes (cfg['type'] == 'other')
                    try:
                        data[field_name] = jnp.zeros(target_shape, dtype=cfg["gen_dtype"])
                    except TypeError:
                        raise NotImplementedError(
                            f"Random generation for dtype {cfg['gen_dtype']} "
                            f"(field: {field_name}) is not implemented robustly."
                        )
        return cls(**data)

    def padding_as_batch(self, batch_shape: tuple[int, ...]):
        if self.structured_type != StructuredType.BATCHED or len(self.shape.batch) > 1:
            raise ValueError(
                f"Padding as batch operation is only supported for BATCHED structured types with at most 1 batch dimension. "
                f"Current type: {self.structured_type}, Current batch shape: {self.shape.batch}"
            )
        if self.shape.batch == batch_shape:
            return self

        new_default_state = self.default(batch_shape)
        new_default_state = new_default_state.at[:self.shape.batch[0]].set(self)
        return new_default_state

    # add method based on default state
    setattr(cls, "reshape", reshape)
    setattr(cls, "flatten", flatten)
    setattr(cls, "random", classmethod(random))
    setattr(cls, "padding_as_batch", padding_as_batch)
    return cls
