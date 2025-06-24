from typing import Any, Tuple

import jax.numpy as jnp

# Represents a JAX dtype, can be a specific type like jnp.int32 or a more generic jnp.dtype
DType = Any


class Packed:
    """
    Represents a packed integer type with a specific number of bits.

    This is a descriptor used within FieldDescriptor to specify that a field's
    values should be bit-packed into a standard integer type (e.g., uint8, uint32).
    For example, `Packed(2)` indicates that each value is 2 bits, and they can be
    packed together into a larger integer type for storage.
    """

    def __init__(self, bits: int):
        if not isinstance(bits, int) or bits <= 0:
            raise ValueError("num_bits must be a positive integer.")
        if bits > 64:
            # JAX supports up to 64-bit integers.
            raise ValueError("bits cannot exceed 64 for packing.")
        self.bits = bits

    def __repr__(self) -> str:
        return f"Packed({self.bits})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Packed):
            return NotImplemented
        return self.bits == other.bits


class FieldDescriptor:
    """
    A descriptor for fields in an xtructure_dataclass.

    This class is used to define the properties of fields in a dataclass decorated with
    @xtructure_dataclass. It specifies the JAX dtype, shape, and default fill value
    for each field.

    Example usage:
        ```python
        @xtructure_dataclass
        class MyData:
            # A scalar uint8 field
            a: FieldDescriptor[jnp.uint8]

            # A field with shape (1, 2) of uint32 values
            b: FieldDescriptor[jnp.uint32, (1, 2)]

            # A float field with custom fill value
            c: FieldDescriptor(dtype=jnp.float32, fill_value=0.0)

            # A nested xtructure_dataclass field
            d: FieldDescriptor[AnotherDataClass]

            # A field of 2-bit values, to be packed.
            e: FieldDescriptor[Packed(2), (10)]
        ```

    The FieldDescriptor can be used with type annotation syntax using square brackets
    or instantiated directly with the constructor for more explicit parameter naming.
    Describes a field in an xtructure_data class, specifying its JAX dtype,
    a default fill value, and its intrinsic (non-batched) shape.
    This allows for auto-generation of the .default() classmethod.
    """

    def __init__(self, dtype: DType, intrinsic_shape: Tuple[int, ...] = (), fill_value: Any = None):
        """
        Initializes a FieldDescriptor.

        Args:
            dtype: The JAX dtype of the field (e.g., jnp.int32, jnp.float32) or a Packed type.
            fill_value: The default value to fill the field's array with
                        (e.g., -1, 0.0).
            intrinsic_shape: The shape of the field itself, before any batching.
                             Defaults to () for a scalar field.
        """
        self.dtype: DType = dtype
        # Set default fill values based on dtype
        if fill_value is None:
            if isinstance(self.dtype, Packed):
                # For packed types, default to all bits set. This is equivalent to -1
                # for an unsigned integer storage type, which will set all bits to 1.
                self.fill_value = -1
            elif hasattr(dtype, "dataclass"):
                # Handle xtructure_dataclass types
                self.fill_value = fill_value
            elif jnp.issubdtype(dtype, jnp.unsignedinteger):
                # For unsigned integers, use -1 (which wraps to max value)
                self.fill_value = -1
            elif jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.floating):
                # For signed integers and floats, use infinity
                self.fill_value = jnp.inf
            else:
                # For other types, keep None
                self.fill_value = fill_value
        else:
            # Use the explicitly provided fill_value
            self.fill_value = fill_value
        self.intrinsic_shape: Tuple[int, ...] = intrinsic_shape

    def __repr__(self) -> str:
        return (
            f"FieldDescriptor(dtype={self.dtype}, "
            f"fill_value={self.fill_value}, "
            f"intrinsic_shape={self.intrinsic_shape})"
        )

    @classmethod
    def __class_getitem__(cls, item: Any) -> "FieldDescriptor":
        """
        Allows for syntax like FieldDescriptor[dtype, intrinsic_shape, fill_value].
        """
        if isinstance(item, tuple):
            if len(item) == 1:
                return cls(item[0])
            elif len(item) == 2:
                # Assuming item[1] is intrinsic_shape or fill_value.
                # Heuristic: if it's a tuple, it's intrinsic_shape. Otherwise, it could be fill_value.
                # This could be ambiguous. For clarity, users might prefer named args with __init__
                # or a more structured approach if this becomes complex.
                if isinstance(item[1], tuple):
                    return cls(item[0], intrinsic_shape=item[1])
                else:  # Assuming it's a fill_value, and intrinsic_shape is default
                    return cls(item[0], fill_value=item[1])
            elif len(item) == 3:
                return cls(item[0], intrinsic_shape=item[1], fill_value=item[2])
            else:
                raise ValueError(
                    "FieldDescriptor[...] expects 1 to 3 arguments: "
                    "dtype, optional intrinsic_shape, optional fill_value"
                )
        else:
            # Single item is treated as dtype
            return cls(item)


# Example usage (to be placed in your class definitions later):
#
# from xtructure.field_descriptors import FieldDescriptor
#
# @xtructure_data
# class MyData:
#     my_scalar_int: FieldDescriptor[jnp.int32, (), -1]
#     my_vector_float: FieldDescriptor[jnp.float32, (10,), 0.0]
#     my_default_shape_int: FieldDescriptor[jnp.uint8]
#     # ... other fields
#
#     # The .default() method would be auto-generated by @xtructure_data
#     # using these descriptors.
