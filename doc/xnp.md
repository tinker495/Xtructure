# `xtructure_numpy` (`xnp`) Operations

The `xtructure_numpy` module provides JAX-compatible operations for working with `@xtructure_dataclass` instances, offering array-like operations that work seamlessly with structured data.

```python
import jax
import jax.numpy as jnp
from xtructure import xtructure_dataclass, FieldDescriptor
from xtructure import xtructure_numpy as xnp

# Available functions in xnp:
# concat, concatenate (same function), pad, stack, reshape, flatten,
# where, unique_mask, set_as_condition_on_array


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

# 3. Pad dataclasses to target size
padded = xnp.pad(result, target_size=5)
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

# 6. Set values conditionally in arrays (with "last True wins")
original = jnp.zeros(5)
indices = jnp.array([0, 2, 0])  # Note: index 0 appears twice
condition = jnp.array([True, True, True])
values = jnp.array([1.0, 2.0, 3.0])  # Last value (3.0) wins for index 0
result_array = xnp.set_as_condition_on_array(original, indices, condition, values)
print(f"Conditional set result: {result_array}")  # [3.0, 0.0, 2.0, 0.0, 0.0]

# 7. Advanced padding with different modes
data = SimpleData.default(shape=(3,))
data = data.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))

# Constant padding (default)
padded_const = xnp.pad(data, target_size=5, constant_values=99)

# Edge padding (repeat edge values)
padded_edge = xnp.pad(data, target_size=5, mode="edge")

# 8. Reshape and flatten (wrappers for dataclass methods)
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

### **`xnp.pad(dataclass, target_size, axis=0, mode='constant', **kwargs)`**
*   Pads a dataclass to reach a target size along the specified axis.
*   **Input**: An `@xtructure_dataclass` instance.
*   **Parameters**:
    *   `target_size`: Target size (int) or shape (tuple).
    *   `axis`: Axis to pad along (default: 0).
    *   `mode`: Padding mode (default: 'constant'). Supports all `jnp.pad` modes: 'constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect', 'symmetric', 'wrap'.
    *   `**kwargs`: Additional arguments passed to `jnp.pad` (e.g., `constant_values` for 'constant' mode).
*   **Output**: Padded dataclass instance.
*   **Behavior**:
    *   For single dataclasses: Creates batched version by replicating the value (mode='constant') or expanding and padding.
    *   For batched dataclasses: Uses existing `padding_as_batch` method when possible, otherwise applies general padding.
    *   Automatically detects optimal padding strategy based on parameters.
*   **Error**: Raises `ValueError` if target size is smaller than current size.

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

### **`xnp.unique_mask(val, key=None, batch_len=None)`**
*   Creates a boolean mask identifying unique elements in a batched Xtructurable, keeping only the entry with minimum cost for each unique state.
*   **Input**: An `Xtructurable` instance with a `uint32ed` attribute (for hashing).
*   **Parameters**:
    *   `key`: Optional cost/priority array for tie-breaking. Lower costs are preferred. If `None`, returns first occurrence.
    *   `batch_len`: Optional explicit batch length. If `None`, inferred from `val.shape.batch[0]`.
*   **Output**: Boolean mask array where `True` indicates the single, cheapest unique value to keep.
*   **Behavior**:
    *   Uses `uint32ed` attribute to compute hash-based uniqueness via `jnp.unique`.
    *   Without `key`: Returns mask for first occurrence of each unique element.
    *   With `key`:
        *   Groups elements by hash using `jnp.unique` with JIT-compatible sizing.
        *   Finds minimum cost per group using segmented operations.
        *   Uses index-based tie-breaking for equal costs (lower index wins).
        *   Excludes entries with infinite cost (padding/invalid entries).
*   **Error**: Raises `ValueError` if `val` lacks `uint32ed` attribute or key length doesn't match batch_len.

### **`xnp.set_as_condition_on_array(array, indices, condition, values_to_set)`**
*   Sets values in an array at specified indices based on conditions, with "last True wins" semantics for duplicates.
*   **Input**:
    *   `array`: JAX array to modify.
    *   `indices`: Array indices (1D) or tuple of index arrays for advanced/multi-dimensional indexing.
    *   `condition`: Boolean array indicating which indices should be updated.
    *   `values_to_set`: Values to set (scalar or array matching the updates).
*   **Output**: Modified array with same shape as input.
*   **Behavior**:
    *   Only sets values where `condition` is `True`.
    *   For duplicate indices: "last True wins" - uses the latest update in the sequence.
    *   Advanced indexing support: handles tuple indices by flattening/reshaping internally.
    *   Uses `segment_max` with timestamps to efficiently handle duplicate index resolution.
    *   Preserves original array values where no updates occur.

### **`xnp.reshape(dataclass, new_shape)`**
*   Wrapper for the dataclass `reshape` method.
*   **Input**: `@xtructure_dataclass` instance and new batch shape.
*   **Output**: Reshaped dataclass instance.

### **`xnp.flatten(dataclass)`**
*   Wrapper for the dataclass `flatten` method.
*   **Input**: `@xtructure_dataclass` instance.
*   **Output**: Flattened dataclass instance with batch dimensions collapsed.

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
padded_batched = xnp.pad(batched, target_size=16)
```

### **Conditional Processing**
```python
# Process data conditionally
condition = data.value > threshold
processed = xnp.where(condition, expensive_operation(data), data)
```

## Technical Notes

**JAX Compatibility**: All `xnp` operations maintain JAX compatibility and support JIT compilation, making them suitable for high-performance GPU computing scenarios.

**Implementation Details**:
- `unique_mask` uses hash-based grouping with segmented operations for efficient duplicate detection.
- `set_as_condition_on_array` uses `segment_max` with timestamps for "last True wins" duplicate resolution.
- `pad` automatically chooses the optimal padding strategy based on input parameters.
- `where` automatically detects dataclass vs scalar arguments for appropriate field-wise operations.
