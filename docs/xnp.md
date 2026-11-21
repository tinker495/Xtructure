# `xtructure.numpy` (`xnp`) Operations

`xtructure.numpy` provides JAX-compatible array operations that preserve `Xtructurable` structure and validate invariants across inputs and outputs.

```python
import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable
from xtructure import numpy as xnp


class SimpleData(Xtructurable):
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    value: FieldDescriptor.scalar(dtype=jnp.float32)


class VectorData(Xtructurable):
    position: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))
    velocity: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


d1 = SimpleData(id=jnp.uint32(1), value=jnp.float32(1.0))
d2 = SimpleData(id=jnp.uint32(2), value=jnp.float32(2.0))
d3 = SimpleData(id=jnp.uint32(1), value=jnp.float32(5.0))

# Combine and pad along batch axes
batch = xnp.concatenate([d1, d2, d3])  # shape (3,)
stacked = xnp.stack([d1, d2])  # shape (2,)
padded = xnp.pad(batch, (0, 2))  # shape (5,)

# Deduplicate keeping lowest cost per unique item
mask = xnp.unique_mask(batch, key=batch.value)  # [True, True, False]
selected = xnp.where(mask, batch, SimpleData.default())

# Gather and manipulate batch axes while preserving intrinsic shapes
vecs = VectorData.default((2, 3))
transposed = xnp.transpose(vecs)  # shape (3, 2); position stays (3,)
taken = xnp.take(batch, jnp.array([0, 2]), axis=0)

# Conditional scatter with "first True wins" for duplicate indices
indices = jnp.array([0, 2, 0])
cond = jnp.array([True, True, True])
updates = SimpleData(
    id=jnp.array([10, 20, 30], dtype=jnp.uint32),
    value=jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32),
)
updated = xnp.update_on_condition(batch, indices, cond, updates)
assert int(updated.id[0]) == 10  # first True at index 0 wins
```

## Key Invariants & Semantics

*   **Batch-axis-only invariant**: Shape and axis operations (`concat`/`concatenate`, `stack`, `pad`, `reshape`, `flatten`/`ravel`, `transpose`, `swapaxes`, `expand_dims`, `squeeze`, `repeat`, `split`, `take`, `take_along_axis`, `tile`) operate on shared batch axes while leaving each field's intrinsic axes untouched.
*   **`unique_mask(val, key=None, filled=None, key_fn=None, batch_len=None, return_index=False, return_inverse=False)`**:
    *   Returns a boolean mask marking one representative per distinct element in a batched `Xtructurable`.
    *   When `key` is provided, selects the representative with the minimum `key` cost (breaking ties by lowest batch index and excluding `jnp.inf` entries when `filled` is omitted). When `filled` is provided, inactive rows are never selected.
    *   By default (`key_fn=None`), hashes `val.uint32ed` to a fixed-width key and deduplicates via `jnp.lexsort`. Custom `key_fn` values of non-`uint32` dtype fall back to exact `jnp.unique`.
    *   Setting `return_index=True` or `return_inverse=True` returns fixed-shape JIT-safe index arrays alongside the mask.
*   **`update_on_condition(dataclass_instance, indices, condition, values_to_set, *, unique_indices=False)`**:
    *   Applies updates only where `condition` is `True`. When duplicate indices appear with `condition=True`, the earliest `True` element wins ("first True wins").
    *   Pass `unique_indices=True` when the caller guarantees no index is selected more than once by `condition`. That unlocks a single masked scatter and avoids allocating the `(len(arr) + 1,)` conflict-resolution buffer per call.
*   **`where_no_broadcast(condition, x, y)`**:
    *   Strict variant of `where` that rejects implicit shape broadcasting and implicit dtype promotion. All operands must match in dataclass tree structure, exact per-field shapes, and exact dtypes (`x` vs `y`), raising `ValueError` or `TypeError` immediately instead of silently broadcasting mismatches.
