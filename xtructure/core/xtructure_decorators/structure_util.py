from typing import Type, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from xtructure.core.field_descriptors import FieldDescriptor, get_field_descriptors
from xtructure.core.type_utils import is_xtructure_dataclass_type

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

    field_descriptors: dict[str, FieldDescriptor] = get_field_descriptors(cls)

    # Pre-calculate generation configurations for the random method
    _field_generation_configs = []
    # Ensure consistent order for key splitting, matching __annotations__
    _field_names_for_random = list(field_descriptors.keys())

    for field_name_cfg in _field_names_for_random:
        descriptor = field_descriptors[field_name_cfg]
        # Retrieve the dtype (or nested xtructure class type) for the current field.
        actual_dtype_or_nested_dtype_tuple = descriptor.dtype
        cfg = {
            "name": field_name_cfg,
            # Keep as-is to preserve historical behavior; normalize to tuple at use sites.
            "default_field_shape": descriptor.intrinsic_shape,
        }

        if is_xtructure_dataclass_type(actual_dtype_or_nested_dtype_tuple):
            # This field is a nested xtructure_data instance
            cfg["type"] = "xtructure"
            # Store the actual nested class type (e.g., Parent, Current)
            cfg["nested_class_type"] = actual_dtype_or_nested_dtype_tuple
        else:
            # This field is a regular JAX array
            actual_dtype = (
                actual_dtype_or_nested_dtype_tuple  # It's a single JAX dtype here
            )
            cfg["actual_dtype"] = actual_dtype  # Store the single JAX dtype

            if jnp.issubdtype(actual_dtype, jnp.integer):
                cfg["type"] = (
                    "bits_int"  # Unified type for all full-range integers via bits
                )
                if jnp.issubdtype(actual_dtype, jnp.unsignedinteger):
                    cfg["bits_gen_dtype"] = (
                        actual_dtype  # Generate bits of this same unsigned type
                    )
                    cfg["view_as_signed"] = False
                else:  # It's a signed integer
                    unsigned_equivalent_str = (
                        f"uint{np.dtype(actual_dtype).itemsize * 8}"
                    )
                    cfg["bits_gen_dtype"] = jnp.dtype(
                        unsigned_equivalent_str
                    )  # Generate bits of corresponding unsigned type
                    cfg["view_as_signed"] = (
                        True  # And then view them as the actual signed type
                    )
            elif jnp.issubdtype(actual_dtype, jnp.floating):
                cfg["type"] = "float"
                cfg["gen_dtype"] = actual_dtype
            elif actual_dtype == jnp.bool_:
                cfg["type"] = "bool"
            else:
                cfg["type"] = "other"  # Fallback
                cfg["gen_dtype"] = actual_dtype
        _field_generation_configs.append(cfg)

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
                current_default_shape = cfg["default_field_shape"]
                target_shape = shape + current_default_shape
                data[field_name] = nested_class.random(
                    shape=target_shape, key=field_key
                )
            else:
                current_default_shape = cfg["default_field_shape"]
                if not isinstance(current_default_shape, tuple):
                    current_default_shape = (current_default_shape,)

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
                        field_key, shape=target_shape
                    )
                else:
                    try:
                        data[field_name] = jnp.zeros(
                            target_shape, dtype=cfg["gen_dtype"]
                        )
                    except TypeError:
                        raise NotImplementedError(
                            f"Random generation for dtype {cfg['gen_dtype']} "
                            f"(field: {field_name}) is not implemented robustly."
                        )
        return cls(**data)

    setattr(cls, "random", classmethod(random))

    # Note: reshape, flatten, transpose, swapaxes, moveaxis, squeeze, expand_dims,
    # roll, flip, rot90, broadcast_to, astype, pad, vstack, etc. are now added by
    # add_xnp_instance_methods() in method_factory.py

    return cls
