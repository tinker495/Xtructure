# `Queue` Usage

A JAX-compatible batched FIFO Queue optimized for parallel GPU execution.

Define stored values by inheriting from `Xtructurable`.

```python
import jax.numpy as jnp
from xtructure import FieldDescriptor, Queue, Xtructurable


class Point(Xtructurable):
    x: FieldDescriptor.scalar(dtype=jnp.uint32)
    y: FieldDescriptor.scalar(dtype=jnp.uint32)


queue = Queue.build(max_size=1000, value_class=Point)

p1 = Point(x=jnp.uint32(1), y=jnp.uint32(2))
queue = queue.enqueue(p1)

batch_points = Point(
    x=jnp.arange(10, dtype=jnp.uint32),
    y=jnp.arange(10, 20, dtype=jnp.uint32),
)
queue = queue.enqueue(batch_points)

front = queue.peek()
assert queue.size == 11

queue, dequeued_batch = queue.dequeue(5)
queue, dequeued_one = queue.dequeue()
queue = queue.clear()
```

## Key `Queue` Contracts

*   **Pure functional return contract**: `Queue` is immutable under JAX transformations. Methods (`enqueue`, `dequeue`, `clear`) never mutate in place—always bind the returned `Queue` (or `(updated_queue, items)` tuple from `dequeue`).
*   **Static config fields**: `Queue` is a `@base_dataclass` with `static_fields=("max_size", "value_class")`, so `max_size` is treated as static compile-time metadata under JIT. Keep `max_size` as a hashable Python `int` rather than a JAX array.
*   **Single packed buffer**: Values are stored as a fixed-shape `uint8[max_size, row_bytes]` buffer initialized with `value_class` default rows so every update executes as one contiguous slice write regardless of PyTree leaf count.
