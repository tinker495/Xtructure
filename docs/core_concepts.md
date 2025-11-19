# Core Concepts: Defining Custom Data Structures

Before using `HashTable` or `BGPQ` in xtructure, you often need to define the structure of the data you want to store. This is done using the `@xtructure_dataclass` decorator and `FieldDescriptor`.

```python
import jax
import jax.numpy as jnp
from xtructure import xtructure_dataclass, FieldDescriptor


# Example: Defining a data structure for HashTable values
@xtructure_dataclass
class MyDataValue:
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    position: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))  # A 3-element vector
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(4,))  # A 4-element boolean array


# Example: Defining a data structure for BGPQ values
@xtructure_dataclass
class MyHeapItem:
    task_id: FieldDescriptor.scalar(dtype=jnp.int32)
    payload: FieldDescriptor.tensor(dtype=jnp.float64, shape=(2, 2))  # A 2x2 matrix
```

## `@xtructure_dataclass`

This decorator transforms a Python class into a JAX-compatible structure and adds several helpful methods and properties:

*   **`shape`** (property): Returns a namedtuple showing the JAX shapes of all fields.
*   **`dtype`** (property): Returns a namedtuple showing the JAX dtypes of all fields.
*   **`__getitem__(self, index)`**: Allows indexing or slicing an instance (e.g., `my_data_instance[0]`). The operation is applied to each field.
*   **`__len__(self)`**: Returns the size of the first dimension of the *first* field, typically used for batch size.
*   **`default(cls, shape=())`** (classmethod): Creates an instance with default values for all fields.
    *   The optional `shape` argument (e.g., `(10,)` or `(5, 2)`) creates a "batched" instance. This means the provided `shape` tuple is prepended to the `intrinsic_shape` of each field defined in the dataclass.
        *   For example, if a field is `data: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))` (intrinsic shape `(3,)`):
            *   Calling `YourClass.default()` or `YourClass.default(shape=())` results in `instance.data.shape` being `(3,)`.
            *   Calling `YourClass.default(shape=(10,))` results in `instance.data.shape` being `(10, 3)`.
            *   Calling `YourClass.default(shape=(5, 2))` results in `instance.data.shape` being `(5, 2, 3)`.
        *   Each field in the instance will be filled with its default value, tiled or broadcasted to this new batched shape.
    *   This method is auto-generated based on `FieldDescriptor` definitions if not explicitly provided.
*   **`random(cls, shape=(), key: jax.random.PRNGKey = ...)`** (classmethod): Creates an instance with random data.
    *   `shape`: Specifies batch dimensions (e.g., `(10,)` or `(5, 2)`), which are prepended to the `intrinsic_shape` of each field.
        *   For example, if a field is `data: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))` (intrinsic shape `(3,)`):
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
*   `check_invariants(self)`: Manually triggers validation logic.
    *   Checks if all fields match their declared `dtype` and `intrinsic_shape`.
    *   Runs any custom `validator` callbacks defined in `FieldDescriptor`.
    *   Automatically called after `__init__` if `validate=True` was passed to the decorator.

## `FieldDescriptor`

Defines the type and shape of each field within an `@xtructure_dataclass`.

*   **Syntax**:
    *   **Factory Methods (Recommended)**:
        *   `FieldDescriptor.tensor(dtype=..., shape=..., fill_value=..., validator=...)`: Explicitly define a tensor field.
        *   `FieldDescriptor.scalar(dtype=..., default=..., validator=...)`: Explicitly define a scalar field.
    *   **Legacy/Direct**:
        *   `FieldDescriptor[jax_dtype]`
        *   `FieldDescriptor[jax_dtype, intrinsic_shape_tuple]`
        *   `FieldDescriptor(dtype=..., intrinsic_shape=..., fill_value=..., validator=...)`
*   **Parameters**:
    *   `dtype`: The JAX dtype (e.g., `jnp.int32`, `jnp.float32`, `jnp.bool_`). Can also be another `@xtructure_dataclass` type for nesting.
    *   `intrinsic_shape` (optional): A tuple defining the field's shape *excluding* batch dimensions (e.g., `(3,)` for a vector, `(2,2)` for a matrix). Defaults to `()` for a scalar.
    *   `fill_value` (optional): The value used when `cls.default()` is called.
        *   Defaults: maximum representable value for unsigned integers, `jnp.inf` for signed integers and floats. `None` for nested structures.
    *   `validator` (optional): A callable that takes the field value and raises an exception (like `ValueError`) if the value is invalid.

### Example: Factory Methods

```python
@xtructure_dataclass
class BetterSyntax:
    # Clear scalar definition
    count: FieldDescriptor.scalar(dtype=jnp.int32, default=0)

    # Clear tensor definition
    points: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,), fill_value=0.0)

    # With custom validation
    prob: FieldDescriptor.scalar(
        dtype=jnp.float32,
        validator=lambda x: 0.0 <= x <= 1.0 or raise_error("Must be probability")
    )
```

### Choosing between legacy and Annotated syntax

Use whichever option fits your tooling. Static analyzers (pyright, mypy, IDEs) often prefer the
`typing.Annotated` form because it exposes the actual runtime type while retaining descriptor metadata.

```python
from typing import Annotated
import jax.numpy as jnp
from xtructure import FieldDescriptor, xtructure_dataclass


@xtructure_dataclass
class LegacySyntax:
    # Field type appears to the type-checker as FieldDescriptor
    value: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


@xtructure_dataclass
class AnnotatedSyntax:
    # Field type is seen as jnp.ndarray by IDEs, metadata comes from FieldDescriptor
    value: Annotated[jnp.ndarray, FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))]
```

Both styles produce identical runtime behavior, so feel free to mix them as you incrementally migrate older
code toward the Annotated form.

### Nested structures

Descriptors can point to another `@xtructure_dataclass`, enabling deeply nested shapes without writing
custom initialization logic. Each nested field uses its own `.default()` for sentinel values, and batch
shapes flow recursively.

```python
@xtructure_dataclass
class SimpleData:
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    value: FieldDescriptor.scalar(dtype=jnp.float32)


@xtructure_dataclass
class Container:
    # Nested dataclasses get their own descriptor
    simple: FieldDescriptor.scalar(dtype=SimpleData)
    history: FieldDescriptor.tensor(dtype=jnp.float32, shape=(4,))

# Automatically builds nested defaults
instance = Container.default(shape=(8,))
assert instance.simple.value.shape == (8,)
```

### Custom defaults via `fill_value_factory`

When the default sentinel depends on the requested batch shape (e.g., NaNs for floats or structured masks),
use `fill_value_factory`. The callable receives `(field_shape, dtype)` and returns the array or value used
by `Container.default(shape=...)`.

```python
def nan_fill(field_shape, dtype):
    return jnp.full(field_shape, jnp.nan, dtype=dtype)


@xtructure_dataclass
class WithFactory:
    metrics: FieldDescriptor.tensor(
        dtype=jnp.float32,
        shape=(3,),
        fill_value_factory=nan_fill,
    )
```

### Runtime validation mode

Pass `validate=True` to `@xtructure_dataclass` to opt into runtime checks that ensure every field matches its
descriptor’s dtype and trailing shape, and satisfies any custom validators. Validation runs after each initialization
(including user-defined `__post_init__` logic).

```python
@xtructure_dataclass(validate=True)
class StrictData:
    vector: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))
    positive: FieldDescriptor.scalar(
        dtype=jnp.int32,
        validator=lambda x: x > 0 or raise_error("Must be positive")
    )

StrictData(vector=jnp.ones((5, 3), dtype=jnp.float32), positive=10)  # OK

# Raises TypeError: StrictData.vector expected dtype float32, got int32
StrictData(vector=jnp.ones((5, 3), dtype=jnp.int32), positive=10)

# Raises error from validator: Must be positive
StrictData(vector=jnp.ones((5, 3), dtype=jnp.float32), positive=-1)
```

You can also trigger validation manually at any time using `.check_invariants()`:

```python
data = StrictData(..., validate=False) # Skip checks during init
# ... modify data ...
data.check_invariants() # Verify consistency now
```

Validation is optional to avoid runtime cost when deserializing known-good data, but it is extremely helpful
while iterating on new structures or integrating external inputs.

## SoA storage with AoS ergonomics

For a detailed explanation of how Xtructure pairs Structure-of-Arrays storage
with Array-of-Structures ergonomics—including the supporting decorators and
common utility patterns—see
[Structure Layout Flexibility](./layout_flexibility.md).
