# Core Concepts: Defining Custom Data Structures

Before using `HashTable` or `BGPQ` in Xtructure, you often need to define the structure of the data you want to store. This is done using the `@xtructure_dataclass` decorator and `FieldDescriptor`.

```python
import jax
import jax.numpy as jnp
from Xtructure import xtructure_dataclass, FieldDescriptor, KEY_DTYPE

# Example: Defining a data structure for HashTable values
@xtructure_dataclass
class MyDataValue:
    id: FieldDescriptor[jnp.uint32]
    position: FieldDescriptor[jnp.float32, (3,)] # A 3-element vector
    flags: FieldDescriptor[jnp.bool_, (4,)]    # A 4-element boolean array

# Example: Defining a data structure for BGPQ values
@xtructure_dataclass
class MyHeapItem:
    task_id: FieldDescriptor[jnp.int32]
    payload: FieldDescriptor[jnp.float64, (2, 2)] # A 2x2 matrix
```

## `@xtructure_dataclass`

This decorator transforms a Python class into a JAX-compatible structure (specifically, a `chex.dataclass`) and adds several helpful methods and properties:

*   **`shape`** (property): Returns a namedtuple showing the JAX shapes of all fields.
*   **`dtype`** (property): Returns a namedtuple showing the JAX dtypes of all fields.
*   **`__getitem__(self, index)`**: Allows indexing or slicing an instance (e.g., `my_data_instance[0]`). The operation is applied to each field.
*   **`__len__(self)`**: Returns the size of the first dimension of the *first* field, typically used for batch size.
*   **`default(cls, shape=())`** (classmethod): Creates an instance with default values for all fields.
    *   The optional `shape` argument (e.g., `(10,)`) creates a batched instance where each field is batched according to this shape, prepended to its intrinsic shape.
    *   This method is auto-generated based on `FieldDescriptor` definitions if not explicitly provided.
*   **`random(cls, shape=(), key: jax.random.PRNGKey = ...)`** (classmethod): Creates an instance with random data.
    *   `shape`: Specifies batch dimensions, prepended to intrinsic field shapes.
    *   `key`: A JAX PRNG key is required for random number generation.
*   `structured_type` (property): An enum (`StructuredType.SINGLE`, `StructuredType.BATCHED`, `StructuredType.UNSTRUCTURED`) indicating instance structure relative to its default.
*   `batch_shape` (property): Shape of batch dimensions if `structured_type` is `BATCHED`.
*   `reshape(self, new_shape)`: Reshapes batch dimensions.
*   `flatten(self)`: Flattens batch dimensions.
*   `__str__(self)` / `str(self)`: Provides a string representation.

## `FieldDescriptor`

Defines the type and shape of each field within an `@xtructure_dataclass`.

*   **Syntax**:
    *   `field_name: FieldDescriptor[jax_dtype]`
    *   `field_name: FieldDescriptor[jax_dtype, intrinsic_shape_tuple]`
    *   `field_name: FieldDescriptor[jax_dtype, intrinsic_shape_tuple, default_fill_value]`
    *   Or direct instantiation: `FieldDescriptor(dtype=..., intrinsic_shape=..., fill_value=...)`
*   **Parameters**:
    *   `dtype`: The JAX dtype (e.g., `jnp.int32`, `jnp.float32`, `jnp.bool_`). Can also be another `@xtructure_dataclass` type for nesting.
    *   `intrinsic_shape` (optional): A tuple defining the field's shape *excluding* batch dimensions (e.g., `(3,)` for a vector, `(2,2)` for a matrix). Defaults to `()` for a scalar.
    *   `fill_value` (optional): The value used when `cls.default()` is called.
        *   Defaults: `-1` (max value) for unsigned integers, `jnp.inf` for signed integers and floats. `None` for nested structures (their own default applies). 