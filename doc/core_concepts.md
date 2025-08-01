# Core Concepts: Defining Custom Data Structures

Before using `HashTable` or `BGPQ` in xtructure, you often need to define the structure of the data you want to store. This is done using the `@xtructure_dataclass` decorator and `FieldDescriptor`.

```python
import jax
import jax.numpy as jnp
from xtructure import xtructure_dataclass, FieldDescriptor


# Example: Defining a data structure for HashTable values
@xtructure_dataclass
class MyDataValue:
    id: FieldDescriptor[jnp.uint32]
    position: FieldDescriptor[jnp.float32, (3,)]  # A 3-element vector
    flags: FieldDescriptor[jnp.bool_, (4,)]  # A 4-element boolean array


# Example: Defining a data structure for BGPQ values
@xtructure_dataclass
class MyHeapItem:
    task_id: FieldDescriptor[jnp.int32]
    payload: FieldDescriptor[jnp.float64, (2, 2)]  # A 2x2 matrix
```

## `@xtructure_dataclass`

This decorator transforms a Python class into a JAX-compatible structure and adds several helpful methods and properties:

*   **`shape`** (property): Returns a namedtuple showing the JAX shapes of all fields.
*   **`dtype`** (property): Returns a namedtuple showing the JAX dtypes of all fields.
*   **`__getitem__(self, index)`**: Allows indexing or slicing an instance (e.g., `my_data_instance[0]`). The operation is applied to each field.
*   **`__len__(self)`**: Returns the size of the first dimension of the *first* field, typically used for batch size.
*   **`default(cls, shape=())`** (classmethod): Creates an instance with default values for all fields.
    *   The optional `shape` argument (e.g., `(10,)` or `(5, 2)`) creates a "batched" instance. This means the provided `shape` tuple is prepended to the `intrinsic_shape` of each field defined in the dataclass.
        *   For example, if a field is `data: FieldDescriptor[jnp.float32, (3,)]` (intrinsic shape `(3,)`):
            *   Calling `YourClass.default()` or `YourClass.default(shape=())` results in `instance.data.shape` being `(3,)`.
            *   Calling `YourClass.default(shape=(10,))` results in `instance.data.shape` being `(10, 3)`.
            *   Calling `YourClass.default(shape=(5, 2))` results in `instance.data.shape` being `(5, 2, 3)`.
        *   Each field in the instance will be filled with its default value, tiled or broadcasted to this new batched shape.
    *   This method is auto-generated based on `FieldDescriptor` definitions if not explicitly provided.
*   **`random(cls, shape=(), key: jax.random.PRNGKey = ...)`** (classmethod): Creates an instance with random data.
    *   `shape`: Specifies batch dimensions (e.g., `(10,)` or `(5, 2)`), which are prepended to the `intrinsic_shape` of each field.
        *   For example, if a field is `data: FieldDescriptor[jnp.float32, (3,)]` (intrinsic shape `(3,)`):
            *   Calling `YourClass.random(key=k)` or `YourClass.random(shape=(), key=k)` results in `instance.data.shape` being `(3,)`.
            *   Calling `YourClass.random(shape=(10,), key=k)` results in `instance.data.shape` being `(10, 3)`.
            *   Calling `YourClass.random(shape=(5, 2), key=k)` results in `instance.data.shape` being `(5, 2, 3)`.
        *   Each field will be filled with random values according to its JAX dtype, and the field arrays will have these new batched shapes.
    *   `key`: A JAX PRNG key is required for random number generation.
*   `structured_type` (property): An enum (`StructuredType.SINGLE`, `StructuredType.BATCHED`, `StructuredType.UNSTRUCTURED`) indicating instance structure relative to its default.
*   `batch_shape` (property): Shape of batch dimensions if `structured_type` is `BATCHED`.
*   `reshape(self, new_shape)`: Reshapes batch dimensions.
*   `flatten(self)`: Flattens batch dimensions.
*   `__str__(self)` / `str(self)`: Provides a string representation.
    *   Handles instances based on their `structured_type`:
        *   `SINGLE`: Uses the original `__str__` method of the instance or a custom pretty formatter for a detailed field-by-field view.
        *   `BATCHED`: For small batches, all items are formatted. For large batches (controlled by `MAX_PRINT_BATCH_SIZE` and `SHOW_BATCH_SIZE`), it provides a summarized view showing the first few and last few elements, along with the batch shape, using `tabulate` for neat formatting.
        *   `UNSTRUCTURED`: Indicates that the data is unstructured relative to its default shape.
*   `default_shape` (property): Returns a namedtuple showing the JAX shapes of all fields as they would be in an instance created by `cls.default()_` (i.e., without any batch dimensions).
*   `at[index_or_slice]` (property): Provides access to an updater object for out-of-place modifications of the instance's fields at the given `index_or_slice`.
    *   `set(values_to_set)`: Returns a new instance with the fields at the specified `index_or_slice` updated with `values_to_set`. If `values_to_set` is an instance of the same dataclass, corresponding fields are used for the update; otherwise, `values_to_set` is applied to all selected field slices.
    *   `set_as_condition(condition, value_to_conditionally_set)`: Returns a new instance where fields at the specified `index_or_slice` are updated based on a JAX boolean `condition`. If an element in `condition` is true, the corresponding element in the field slice is updated with `value_to_conditionally_set`.
*   `save(self, path)`: Saves the instance to a file.
    *   `path`: File path where the instance will be saved (typically with `.npz` extension).
    *   The instance is serialized and saved using the xtructure IO module.
*   `load(cls, path)` (classmethod): Loads an instance from a file.
    *   `path`: File path from which to load the instance.
    *   Returns an instance of the class loaded from the specified file.
    *   Raises a `TypeError` if the loaded instance is not of the expected class type.

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
