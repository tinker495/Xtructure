# `Stack` Usage

A JAX-compatible batched LIFO Stack optimized for parallel GPU execution.

Define stored values by inheriting from `Xtructurable`.

```python
import jax.numpy as jnp
from xtructure import FieldDescriptor, Stack, Xtructurable


class Point(Xtructurable):
    x: FieldDescriptor.scalar(dtype=jnp.uint32)
    y: FieldDescriptor.scalar(dtype=jnp.uint32)


stack = Stack.build(max_size=1000, value_class=Point)

p1 = Point(x=jnp.uint32(1), y=jnp.uint32(2))
stack = stack.push(p1)

batch_points = Point(
    x=jnp.arange(10, dtype=jnp.uint32),
    y=jnp.arange(10, 20, dtype=jnp.uint32),
)
stack = stack.push(batch_points)

top = stack.peek()
assert stack.size == 11

stack, popped_batch = stack.pop(5)
stack, popped_one = stack.pop()
```

## Key `Stack` Contracts

*   **Pure functional return contract**: `Stack` is immutable under JAX transformations. Methods (`push`, `pop`) return the updated `Stack` (or `(updated_stack, items)` tuple) rather than mutating in place.
*   **Static config fields**: `Stack` is a `@base_dataclass` with `static_fields=("max_size", "value_class")`, so `max_size` is treated as static compile-time metadata under JIT. Keep `max_size` as a hashable Python `int` rather than a JAX array.
*   **Single packed buffer**: Values are stored as a fixed-shape `uint8[max_size, row_bytes]` buffer initialized with `value_class` default rows so every push executes as one contiguous slice write regardless of PyTree leaf count.
