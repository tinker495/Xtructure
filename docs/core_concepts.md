# Core Concepts: Defining Custom Data Structures

Before using `HashTable` or `BGPQ` in xtructure, you often need to define the structure of the data you want to store. Inherit from `Xtructurable` and declare fields with `FieldDescriptor`.

> **Legacy decorator:** `@xtructure_dataclass` emits `FutureWarning` at class definition.
> New code should use inheritance; see [Legacy Decorator Migration](migration.md) for options,
> warning handling, and unsupported inheritance patterns. `@base_dataclass` remains supported.

```python
import jax
import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable


# Example: Defining a data structure for HashTable values
class MyDataValue(Xtructurable):
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    position: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))  # A 3-element vector
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(4,))  # A 4-element boolean array


# Example: Defining a data structure for BGPQ values
class MyHeapItem(Xtructurable):
    task_id: FieldDescriptor.scalar(dtype=jnp.int32)
    payload: FieldDescriptor.tensor(dtype=jnp.float64, shape=(2, 2))  # A 2x2 matrix
```

## `Xtructurable`

Inheriting from this base transforms a Python class into a JAX-compatible structure and adds several helpful methods and properties:

*   **`shape`** (property): Returns the leading axes shared by every field as a plain tuple, e.g. `(10,)`.
*   **`layout`** (property): Returns instance layout metadata, including each field's intrinsic shape.
*   **`dtype`** (property): Returns a namedtuple showing the JAX dtypes of all fields.
*   **`__getitem__(self, index)`**: Allows indexing or slicing an instance (e.g., `my_data_instance[0]`). The operation is applied to each field.
*   **`__len__(self)`**: Returns the first batch dimension for `BATCHED` instances and 1 for `SINGLE` instances. Raises `TypeError` for `UNSTRUCTURED` instances.
*   **`__iter__(self)`**: Supports `for item in instance` and `list(instance)`, yielding instances of the same class.
    *   `BATCHED`: Iterates exactly `len(instance)` times over the first batch axis. Remaining batch axes stay on each yielded item; an empty first batch axis yields no items.
    *   `SINGLE`: Calling `iter(instance)` raises `TypeError`, even though `len(instance)` is 1.
    *   `UNSTRUCTURED`: Calling `iter(instance)` raises `TypeError`, matching the `len()` contract.
    *   Iteration stops at the batch boundary even when JAX array indexing would clamp an out-of-bounds index. The same behavior applies to legacy decorator-built classes and generated packed/unpacked classes.
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
*   `shape` (property): The leading axes shared by every field: `()` for `SINGLE`, `(N, ...)` for `BATCHED`. `UNSTRUCTURED` instances raise `TypeError`, like `ndim` and `len()`. Field arrays keep their own full shape, e.g. `instance.data.shape == (10, 3)` while `instance.shape == (10,)`.
*   `layout` (property): Instance layout metadata. `instance.layout.field_shape_for("data")` returns a field's intrinsic shape, e.g. `(3,)`.
*   `batch_shape` (property): Same value as `shape` for structured instances; `-1` for `UNSTRUCTURED`.
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
    *   Xtructure `.npz` files should be treated as trusted inputs: loading imports the module recorded in file metadata so the class can be reconstructed.
*   `check_invariants(self)`: Manually triggers validation logic.
    *   Checks if all fields match their declared `dtype` and `intrinsic_shape`.
    *   Runs any custom `validator` callbacks defined in `FieldDescriptor`.
    *   Automatically called after `__init__` if `validate=True` was passed in the class header.

## `FieldDescriptor`

Defines the type and shape of each field within an `Xtructurable` subclass.

*   **Syntax**:
    *   **Factory Methods (Recommended)**:
        *   `FieldDescriptor.tensor(dtype=..., shape=..., fill_value=..., validator=...)`: Explicitly define a tensor field.
        *   `FieldDescriptor.scalar(dtype=..., default=..., validator=...)`: Explicitly define a scalar field.
    *   **Legacy/Direct**:
        *   `FieldDescriptor[jax_dtype]`
        *   `FieldDescriptor[jax_dtype, intrinsic_shape_tuple]`
        *   `FieldDescriptor(dtype=..., intrinsic_shape=..., fill_value=..., validator=...)`
*   **Parameters**:
    *   `dtype`: The JAX dtype (e.g., `jnp.int32`, `jnp.float32`, `jnp.bool_`). Can also be another `Xtructurable` subclass for nesting.
    *   `intrinsic_shape` (optional): A tuple defining the field's shape *excluding* batch dimensions (e.g., `(3,)` for a vector, `(2,2)` for a matrix). Defaults to `()` for a scalar.
    *   `fill_value` (optional): The value used when `cls.default()` is called.
        *   Defaults: maximum representable value for unsigned integers, `jnp.inf` for signed integers and floats. `None` for nested structures.
    *   `validator` (optional): A callable that takes the field value and raises an exception (like `ValueError`) if the value is invalid.
    *   `bits` (optional): Active bits per value for bitpacking (integer in `[1, 32]`).
        *   Used by **aggregate bitpacking** (see below) and by **IO packing** (`save(..., packed=True)`) for that field.

### Example: Factory Methods

```python
class BetterSyntax(Xtructurable):
    # Clear scalar definition
    count: FieldDescriptor.scalar(dtype=jnp.int32, default=0)

    # Clear tensor definition
    points: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,), fill_value=0.0)

    # With custom validation
    prob: FieldDescriptor.scalar(
        dtype=jnp.float32, validator=lambda x: 0.0 <= x <= 1.0 or raise_error("Must be probability")
    )
```

### Descriptor shorthand and Annotated syntax

Use whichever option fits your tooling. Static analyzers (pyright, mypy, IDEs) often prefer the
`typing.Annotated` form because it exposes the actual runtime type while retaining descriptor metadata.

```python
from typing import Annotated
import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable


class ShorthandSyntax(Xtructurable):
    # Field type appears to the type-checker as FieldDescriptor
    value: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


class AnnotatedSyntax(Xtructurable):
    # Field type is seen as jnp.ndarray by IDEs, metadata comes from FieldDescriptor
    value: Annotated[jnp.ndarray, FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))]
```

Descriptor shorthand remains supported; it is separate from the deprecated class decorator.
Both field styles produce identical runtime behavior, so feel free to mix them as you incrementally migrate older
code toward the Annotated form.

### Nested structures

Descriptors can point to another `Xtructurable` subclass, enabling deeply nested shapes without writing
custom initialization logic. Each nested field uses its own `.default()` for sentinel values, and batch
shapes flow recursively.

```python
class SimpleData(Xtructurable):
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    value: FieldDescriptor.scalar(dtype=jnp.float32)


class Container(Xtructurable):
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


class WithFactory(Xtructurable):
    metrics: FieldDescriptor.tensor(
        dtype=jnp.float32,
        shape=(3,),
        fill_value_factory=nan_fill,
    )
```

### Runtime validation mode

Use `class StrictData(Xtructurable, validate=True):` to opt into runtime checks that ensure every field matches its
descriptor’s dtype and trailing shape, and satisfies any custom validators. Validation runs after each initialization
(including user-defined `__post_init__` logic).

```python
class StrictData(Xtructurable, validate=True):
    vector: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))
    positive: FieldDescriptor.scalar(dtype=jnp.int32, validator=lambda x: x > 0 or raise_error("Must be positive"))


StrictData(vector=jnp.ones((5, 3), dtype=jnp.float32), positive=10)  # OK

# Raises TypeError: StrictData.vector expected dtype float32, got int32
StrictData(vector=jnp.ones((5, 3), dtype=jnp.int32), positive=10)

# Raises error from validator: Must be positive
StrictData(vector=jnp.ones((5, 3), dtype=jnp.float32), positive=-1)
```

## In-memory bitpacked fields (memory-optimized runtime state)

`xtructure` also supports **storing fields as packed `uint8` byte streams in-memory**, while exposing an
easy-to-use logical view for observations / transitions.

Use `FieldDescriptor.packed_tensor(...)` to declare the logical **Xtructure Schema** shape
for a field that **Type Layout** stores as packed bytes, then:

- Access `<field>_unpacked` to get the logical array.
- Use `set_unpacked(field=...)` to update from a logical array (it will be packed automatically).
- Use `from_unpacked(...)` to construct a packed instance directly from logical arrays (avoids an extra `.default()`).
- Read storage byte facts from **Packed Field Layout** via `get_type_layout(...)`, not from `FieldDescriptor`.

Example:

```python
import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable


class PuzzleState(Xtructurable):
    # Logical Schema: 6 faces of size*size values, each value in [0, 7] => 3 bits/value.
    # Packed Field Layout derives the stored uint8 byte-stream shape.
    faces: FieldDescriptor.packed_tensor(
        shape=(6, 54),
        packed_bits=3,
    )


state = PuzzleState.default()

# Read logical view (unpacked):
faces = state.faces_unpacked  # shape (6, 54), dtype uint8

# Write logical view (auto-packed into state.faces uint8 byte stream):
state2 = state.set_unpacked(faces=(faces + 1) & jnp.uint8(7))

# Packed-first construction (more direct than `PuzzleState.default().set_unpacked(...)`):
packed = PuzzleState.from_unpacked(faces=faces)
```

Notes:
- `packed_bits` supports `[1, 32]`.
- `shape=` is the logical unpacked **Intrinsic Shape**. The stored field shape is the
  byte-stream **Intrinsic Shape** derived by **Packed Field Layout**.
- If you omit `unpacked_dtype`, **Type Layout** interprets the default **Packed Data Kind**:
  - `bool` for `packed_bits == 1`
  - `uint8` for `2 <= packed_bits <= 8`
  - `uint32` for `packed_bits > 8`

## Aggregate bitpacking across all fields (single byte-stream per instance)

If you want to **pack multiple fields together into one contiguous bitstream** (minimizing per-field padding),
you can enable (or auto-enable) aggregate bitpacking.

Auto-enable rule:
- If **every primitive leaf** declares `bits=...` (nested `Xtructurable` fields supported when the nesting
  field is scalar), aggregate bitpacking is enabled automatically (so you can omit the class option).

### Unified bitpacking policy (`bitpack=...`)

Set `bitpack=...` in the class header, for example `class State(Xtructurable, bitpack="aggregate"):`:

- `bitpack="auto"` (default): enables aggregate packing when all primitive leaves (including scalar-nested leaves)
  declare `bits=...`. Also enables `packed_tensor` accessors.
- `bitpack="aggregate"`: force aggregate packing for the whole dataclass.
- `bitpack="field"`: only enable `FieldDescriptor.packed_tensor(...)` accessors (no aggregate packing).
- `bitpack="off"`: disable in-memory bitpacking helpers on the class.

```python
import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable


class AggState(Xtructurable):
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(17,), bits=1)
    faces: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(6, 54), bits=3)
    codes: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(5,), bits=12)


# Packed-first usage (recommended for memory efficiency):
p = AggState.Packed.default()  # AggStatePacked(words=uint32[...], tail=uint8[0..2])

# Unpack view only when needed (observation/transition):
u = p.unpacked  # AggStateUnpacked view (default dtypes: bool/uint8/uint32)
o = p.as_original()  # AggState reconstructed with declared dtypes (passes validate=True)

# Partial decode: unpack a single field (and optionally only some indices) without materializing all fields.
faces_only = p.unpack_field("faces")  # same as u.faces, but avoids unpacking flags/codes
codes_some = p.unpack_field("codes", indices=[0, 2])  # returns batch + (2,)

# If you already have a logical instance and want to store it compactly:
p2 = AggState.default().packed
```

Aggregate bitpacking also reserves generated module-level names so IO metadata can round-trip generated classes:

- `AggStatePacked` backs `AggState.Packed`.
- `AggStateUnpacked` backs `AggState.Packed.default().unpacked`; nested aggregate views reserve the same
  `<NestedClassName>Unpacked` pattern.
- Re-running or reloading the same aggregate class definition replaces earlier generated classes with the same
  generated role.
- A user-defined object with one of these reserved names is treated as a collision and raises `TypeError` during
  class definition instead of being overwritten silently.

You can also trigger validation manually at any time using `.check_invariants()`:

```python
data = MyDataValue.default()  # Validation is disabled by default
data.check_invariants()  # Verify dtype, intrinsic shape, and custom validators now
```

Validation is optional to avoid runtime cost when deserializing known-good data, but it is extremely helpful
while iterating on new structures or integrating external inputs.

## `base_dataclass` and `static_fields`

`Xtructurable` inheritance uses the same class transformation built on top of `@base_dataclass`.
You can still use `@base_dataclass` directly for "plain" JAX PyTrees that are not SoA-backed
`Xtructurable` structures (for example: `HashTable`, `BGPQ`,
`Queue`, `Stack`).

The key extra feature is `static_fields`:

- `@base_dataclass(static_fields=(...))` marks specific dataclass fields as **static PyTree metadata (aux_data)**.
- Static fields are **not JAX leaves**, so they do not get transferred to device and they participate in JIT cache
  keys (changing them triggers a recompile).
- Static field values must be **Python-hashable** (e.g. `int`, `str`, `tuple`), otherwise JAX tracing will error.

This is the intended way to store configuration values like sizes / capacities / branching factors that are used to
decide shapes or control-flow, instead of:

- threading them through every `jax.jit` call as explicit arguments, or
- re-inferring them from runtime array shapes (e.g. `n = x.shape[0]`) when the value is logically a config constant.

Example:

```python
import jax
import jax.numpy as jnp
from xtructure import base_dataclass


@base_dataclass(static_fields=("bucket_size",))
class AlgoConfig:
    bucket_size: int
    data: jax.Array


@jax.jit
def f(cfg: AlgoConfig):
    # bucket_size is static metadata, so this is a compile-time constant.
    bucket_size = int(cfg.bucket_size)
    return cfg.data.reshape((-1, bucket_size))
```

## SoA storage with AoS ergonomics

For a detailed explanation of how Xtructure pairs Structure-of-Arrays storage
with Array-of-Structures ergonomics—including the class transformation and
common utility patterns—see
[Structure Layout Flexibility](./layout_flexibility.md).
