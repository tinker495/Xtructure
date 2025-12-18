# `Queue` Usage

A JAX-compatible batched Queue data structure, designed for FIFO (First-In, First-Out) operations. It is optimized for parallel execution on hardware like GPUs.

```python
import jax
import jax.numpy as jnp
from xtructure import Queue, xtructure_dataclass, FieldDescriptor


# Define a data structure to store in the queue
@xtructure_dataclass
class Point:
    x: FieldDescriptor.scalar(dtype=jnp.uint32)
    y: FieldDescriptor.scalar(dtype=jnp.uint32)


# 1. Build the Queue
#    Queue.build(max_size, value_class)
queue = Queue.build(max_size=1000, value_class=Point)

# 2. Enqueue a single item
p1 = Point(x=jnp.array(1, dtype=jnp.uint32), y=jnp.array(2, dtype=jnp.uint32))
queue = queue.enqueue(p1)
print(f"Queue size after enqueuing one item: {queue.size}")
print(f"Queue head: {queue.head}, Queue tail: {queue.tail}")


# 3. Enqueue a batch of items
batch_points = Point(x=jnp.arange(10, dtype=jnp.uint32), y=jnp.arange(10, 20, dtype=jnp.uint32))
queue = queue.enqueue(batch_points)
print(f"Queue size after enqueuing a batch: {queue.size}")
print(f"Queue head: {queue.head}, Queue tail: {queue.tail}")

# 4. Peek at the front item
#    Does not modify the queue
peeked_item = queue.peek()
print("Peeked item:", peeked_item)
assert queue.size == 11  # Unchanged

# 5. Dequeue a batch of items
#    Removes the first 5 items from the queue
queue, dequeued_items = queue.dequeue(5)
print(f"Queue size after dequeuing 5 items: {queue.size}")
print(f"Queue head: {queue.head}, Queue tail: {queue.tail}")
print("Dequeued items (x-values):", dequeued_items.x)

# 6. Dequeue a single item
queue, dequeued_item = queue.dequeue()
print(f"Queue size after dequeuing one item: {queue.size}")
print("Dequeued item:", dequeued_item)

# 7. Clear the queue
queue = queue.clear()
print(f"Queue size after clearing: {queue.size}")
print(f"Queue head: {queue.head}, Queue tail: {queue.tail}")
```

## Key `Queue` Details

*   **FIFO Principle**: The first element added to the queue will be the first one to be removed.
*   **API Style**: The methods (`enqueue`, `dequeue`, `clear`) modify the queue's state and return the modified instance, allowing for a chained, functional-style usage pattern.
*   **Static config fields**: `Queue` is a `@base_dataclass` with `static_fields=("max_size",)`, so `max_size` is treated as static metadata under JIT. Keep it as a Python `int` (hashable).

*   **`Queue.build(max_size, value_class)`**:
    *   `max_size` (int): The maximum number of elements the queue can hold.
    *   `value_class` (Xtructurable): The class of the data structure to be stored (e.g., `Point`).

*   **`queue.enqueue(items)`**:
    *   `items` (Xtructurable): An instance or a batch of instances to add to the end of the queue.
    *   Returns the updated `Queue` instance.

*   **`queue.dequeue(num_items=1)`**:
    *   `num_items` (int): The number of items to remove from the front of the queue.
    *   Returns a tuple containing:
        1.  The updated `Queue` instance.
        2.  The `Xtructurable` containing the dequeued items.

*   **`queue.peek(num_items=1)`**:
    *   `num_items` (int): The number of items to view from the front of the queue.
    *   Returns the `Xtructurable` containing the front items without modifying the queue.

*   **`queue.clear()`**:
    *   Resets the `head` and `tail` of the queue to 0, effectively emptying it.
    *   Returns the updated `Queue` instance.
```
