# `xtructure_numpy` (`xnp`) Operations

The `xtructure_numpy` module provides JAX-compatible operations for working with `@xtructure_dataclass` instances, offering array-like operations that work seamlessly with structured data.

```python
import jax
import jax.numpy as jnp
from xtructure import xtructure_dataclass, FieldDescriptor

# New import path available:
from xtructure import numpy as xnp

# Or the traditional way:
from xtructure import xtructure_numpy as xnp

# Available functions in xnp:
# concat, concatenate (same function), pad, stack, reshape, flatten,
# where, unique_mask, take, update_on_condition,
# tile, transpose, swap_axes


# Define example data structures
@xtructure_dataclass
class SimpleData:
    id: FieldDescriptor[jnp.uint32]
    value: FieldDescriptor[jnp.float32]


@xtructure_dataclass
class VectorData:
    position: FieldDescriptor[jnp.float32, (3,)]
    velocity: FieldDescriptor[jnp.float32, (3,)]


# 1. Concatenate dataclasses
data1 = SimpleData.default()
data1 = data1.replace(id=jnp.array(1), value=jnp.array(1.0))
data2 = SimpleData.default()
data2 = data2.replace(id=jnp.array(2), value=jnp.array(2.0))
data3 = SimpleData.default()
data3 = data3.replace(id=jnp.array(3), value=jnp.array(3.0))

# Concatenate single dataclasses into a batch
result = xnp.concatenate([data1, data2, data3])
print(f"Concatenated batch shape: {result.shape.batch}")  # (3,)
print(f"IDs: {result.id}")  # [1, 2, 3]

# 2. Stack dataclasses
stacked = xnp.stack([data1, data2])
print(f"Stacked batch shape: {stacked.shape.batch}")  # (2,)

# 3. Pad dataclasses with specified padding
padded = xnp.pad(result, (0, 2))
print(f"Padded batch shape: {padded.shape.batch}")  # (5,)

# 4. Conditional selection with where
condition = jnp.array([True, False, True])
selected = xnp.where(condition, result[:3], -1)
print(f"Where result IDs: {selected.id}")  # [1, -1, 3]

# 5. Unique mask for filtering duplicates
data_with_dupes = SimpleData.default(shape=(5,))
data_with_dupes = data_with_dupes.replace(id=jnp.array([1, 2, 1, 3, 2]), value=jnp.array([1.0, 2.0, 1.0, 3.0, 2.0]))
unique_mask = xnp.unique_mask(data_with_dupes)
print(f"Unique mask: {unique_mask}")  # [True, True, False, True, False]

# 6. Take elements from specific indices
data = SimpleData.default(shape=(10,))
data = data.replace(id=jnp.arange(10), value=jnp.arange(10, dtype=jnp.float32))
taken = xnp.take(data, jnp.array([0, 2, 4, 6, 8]))
print(f"Taken IDs: {taken.id}")  # [0, 2, 4, 6, 8]

# 7. Update values conditionally with "first True wins" semantics
original = jnp.zeros(5)
indices = jnp.array([0, 2, 0])  # Note: index 0 appears twice
condition = jnp.array([True, True, True])
values = jnp.array([1.0, 2.0, 3.0])  # Last value (3.0) wins for index 0
result_array = xnp.update_on_condition(original, indices, condition, values)
print(f"Conditional update result: {result_array}")  # [3.0, 0.0, 2.0, 0.0, 0.0]

# 8. Advanced padding with different modes
data = SimpleData.default(shape=(3,))
data = data.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))

# Constant padding (default)
padded_const = xnp.pad(data, (0, 2), constant_values=99)

# Edge padding (repeat edge values)
padded_edge = xnp.pad(data, (0, 2), mode="edge")

# 9. Reshape and flatten (wrappers for dataclass methods)
batched_data = SimpleData.default(shape=(6,))
reshaped = xnp.reshape(batched_data, (2, 3))
flattened = xnp.flatten(reshaped)
```

## Key `xnp` Operations

### **`xnp.concatenate(dataclasses, axis=0)` / `xnp.concat(dataclasses, axis=0)`**
*   Concatenates a list of dataclasses along the specified axis (`concatenate` and `concat` are aliases).
*   **Input**: List of `@xtructure_dataclass` instances. All must be of the same type and structured type.
*   **Parameters**:
    *   `dataclasses`: List of dataclass instances to concatenate.
    *   `axis`: Axis along which to concatenate (default: 0).
*   **Output**: A single batched dataclass with `structured_type.name == "BATCHED"`.
*   **Behavior**:
    *   Single dataclasses: Converted to batched (size 1) then concatenated.
    *   Batched dataclasses: Concatenated directly along specified axis.
    *   Validates batch shape compatibility (all dimensions except concat axis must match).
*   **Error**: Raises `ValueError` for empty lists, mixed types, or incompatible structures.

### **`xnp.stack(dataclasses_list, axis=0)`**
*   Stacks dataclasses along a new dimension.
*   **Input**: List of `@xtructure_dataclass` instances with compatible batch shapes.
*   **Parameters**:
    *   `axis` (int): The axis along which to stack. Default is 0.
*   **Output**: A batched dataclass with an additional dimension.
*   **Error**: Raises `ValueError` for empty lists or incompatible batch shapes.

### **`xnp.pad(dataclass, pad_width, mode='constant', **kwargs)`**
*   Pads a dataclass with specified padding widths, following jnp.pad interface.
*   **Input**: An `@xtructure_dataclass` instance.
*   **Parameters**:
    *   `pad_width`: Padding width specification following jnp.pad convention:
        - int: Same padding (before, after) for all axes
        - sequence of int: Padding for each axis (before, after)
        - sequence of pairs: (before, after) padding for each axis
    *   `mode`: Padding mode (default: 'constant'). Supports all `jnp.pad` modes: 'constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect', 'symmetric', 'wrap'.
    *   `**kwargs`: Additional arguments passed to `jnp.pad` (e.g., `constant_values` for 'constant' mode).
*   **Output**: Padded dataclass instance.
*   **Behavior**:
    *   For single dataclasses: Creates batched version by applying padding to create new batch dimension.
    *   For batched dataclasses: Uses existing `padding_as_batch` method when possible, otherwise applies general padding.
    *   Automatically detects optimal padding strategy based on parameters.
*   **Error**: Raises `ValueError` if pad_width is incompatible with dataclass structure.

### **`xnp.where(condition, x, y)`**
*   Conditional selection for dataclasses, similar to `jnp.where`.
*   **Input**:
    *   `condition`: Boolean array or scalar.
    *   `x`: `@xtructure_dataclass` instance (Xtructurable).
    *   `y`: `@xtructure_dataclass` instance or scalar/array.
*   **Output**: Dataclass with values selected based on condition.
*   **Behavior**:
    *   Element-wise selection where `condition` is `True` → `x`, `False` → `y`.
    *   Automatically detects if `y` is a dataclass (multiple tree leaves or has `__dataclass_fields__`) or scalar.
    *   If `y` is a dataclass: applies `jnp.where` field-wise between `x` and `y`.
    *   If `y` is a scalar: applies `jnp.where` between each field of `x` and the scalar `y`.

### **`xnp.unique_mask(val, key=None, batch_len=None, return_index=False, return_inverse=False)`**
*   Creates a boolean mask identifying unique elements in a batched Xtructurable, keeping only the entry with minimum cost for each unique state.
*   **Input**: An `Xtructurable` instance with a `uint32ed` attribute (for hashing).
*   **Parameters**:
    *   `key`: Optional cost/priority array for tie-breaking. Lower costs are preferred. If `None`, returns first occurrence.
    *   `batch_len`: Optional explicit batch length. If `None`, inferred from `val.shape.batch[0]`.
    *   `return_index`: If `True`, also return indices of unique elements.
    *   `return_inverse`: If `True`, also return inverse indices for reconstructing original array.
*   **Output**: Boolean mask array where `True` indicates the single, cheapest unique value to keep. If `return_index` or `return_inverse` is `True`, returns a tuple.
*   **Behavior**:
    *   Uses `uint32ed` attribute to compute hash-based uniqueness via `jnp.unique`.
    *   Without `key`: Returns mask for first occurrence of each unique element.
    *   With `key`:
        *   Groups elements by hash using `jnp.unique` with JIT-compatible sizing.
        *   Finds minimum cost per group using segmented operations.
        *   Uses index-based tie-breaking for equal costs (lower index wins).
        *   Excludes entries with infinite cost (padding/invalid entries).
*   **Error**: Raises `ValueError` if `val` lacks `uint32ed` attribute or key length doesn't match batch_len.

### **`xnp.take(dataclass_instance, indices, axis=0)`**
*   Takes elements from a dataclass along the specified axis, similar to `jnp.take`.
*   **Input**:
    *   `dataclass_instance`: The dataclass instance to take elements from.
    *   `indices`: Array of indices to take.
    *   `axis`: Axis along which to take elements (default: 0).
*   **Output**: A new dataclass instance with elements taken from the specified indices.
*   **Behavior**:
    *   Applies `jnp.take` to each field of the dataclass.
    *   Maintains the structure and field relationships of the original dataclass.
    *   Works with both single and batched dataclasses.
*   **Examples**:
    ```python
    # Take specific elements from a batched dataclass
    data = MyData.default((5,))
    result = xnp.take(data, jnp.array([0, 2, 4]))
    # result will have batch shape (3,) with elements at indices 0, 2, 4

    # Take elements along a different axis
    data = MyData.default((3, 4))
    result = xnp.take(data, jnp.array([1, 3]), axis=1)
    # result will have batch shape (3, 2) with elements at indices 1, 3 along axis 1
    ```

### **`xnp.update_on_condition(dataclass_instance, indices, condition, values_to_set)`**
*   Updates values in a dataclass based on a condition, ensuring "first True wins" for duplicate indices.
*   **Input**:
    *   `dataclass_instance`: The dataclass instance to update.
    *   `indices`: Indices where updates should be applied (1D array or tuple for advanced indexing).
    *   `condition`: Boolean array indicating which updates should be applied.
    *   `values_to_set`: Values to set when condition is True. Can be a dataclass instance (compatible with dataclass_instance) or a scalar value.
*   **Output**: A new dataclass instance with updated values.
*   **Behavior**:
    *   Only sets values where `condition` is `True`.
    *   For duplicate indices: "first True wins" - uses the first update in the sequence.
    *   Advanced indexing support: handles tuple indices by flattening/reshaping internally.
    *   Automatically detects if `values_to_set` is a dataclass or scalar.
    *   If `values_to_set` is a dataclass: applies update field-wise between dataclasses.
    *   If `values_to_set` is a scalar: applies the scalar value to all fields.
*   **Examples**:
    ```python
    # Update with scalar value
    updated = xnp.update_on_condition(dataclass, indices, condition, -1)

    # Update with another dataclass
    updated = xnp.update_on_condition(dataclass, indices, condition, new_values)
    ```

### **`xnp.reshape(dataclass, new_shape)`**
*   Wrapper for the dataclass `reshape` method.
*   **Input**: `@xtructure_dataclass` instance and new batch shape.
*   **Output**: Reshaped dataclass instance.

### **`xnp.flatten(dataclass)`**
*   Wrapper for the dataclass `flatten` method.
*   **Input**: `@xtructure_dataclass` instance.
*   **Output**: Flattened dataclass instance with batch dimensions collapsed.

## Import Options

You can import the xtructure_numpy functionality in several ways:

```python
# New recommended import path:
from xtructure import numpy as xnp

# Traditional import path:
from xtructure import xtructure_numpy as xnp

# Direct import:
import xtructure.xtructure_numpy as xnp
```

## Usage Patterns

### **Filtering and Deduplication**
```python
# Remove duplicates using unique_mask
data = SimpleData.default(shape=(100,))
# ... populate data ...
costs = jnp.array([...])  # Lower costs preferred
unique_mask = xnp.unique_mask(data, key=costs)
filtered_data = xnp.where(unique_mask, data, SimpleData.default())
```

### **Batching Operations**
```python
# Combine multiple single dataclasses
singles = [SimpleData.default() for _ in range(10)]
# ... populate singles ...
batched = xnp.concatenate(singles)

# Pad to fixed size for uniform batching
padded_batched = xnp.pad(batched, (0, 6))  # Assuming batched has size 10
```

### **Conditional Processing**
```python
# Process data conditionally
condition = data.value > threshold
processed = xnp.where(condition, expensive_operation(data), data)
```

### **Selective Element Access**
```python
# Take specific elements from a dataset
important_indices = jnp.array([0, 5, 10, 15])
important_data = xnp.take(dataset, important_indices)
```

### **Conditional Updates**
```python
# Update specific elements based on conditions
indices = jnp.array([1, 3, 5])
condition = jnp.array([True, False, True])
new_values = MyData.default(shape=(3,))
updated_data = xnp.update_on_condition(data, indices, condition, new_values)
```

## Technical Notes

**JAX Compatibility**: All `xnp` operations maintain JAX compatibility and support JIT compilation, making them suitable for high-performance GPU computing scenarios.

**Implementation Details**:
- `unique_mask` uses hash-based grouping with segmented operations for efficient duplicate detection.
- `update_on_condition` uses `segment_max` with timestamps for "first True wins" duplicate resolution.
- `pad` automatically chooses the optimal padding strategy based on input parameters.
- `where` automatically detects dataclass vs scalar arguments for appropriate field-wise operations.
- `take` applies `jnp.take` to each field while maintaining dataclass structure.
