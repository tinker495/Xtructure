# `Stack` Usage

A JAX-compatible batched Stack data structure, designed for LIFO (Last-In, First-Out) operations. It is optimized for parallel execution on hardware like GPUs.

```python
import jax
import jax.numpy as jnp
from xtructure import Stack, xtructure_dataclass, FieldDescriptor


# Define a data structure to store in the stack
@xtructure_dataclass
class Point:
    x: FieldDescriptor[jnp.uint32]
    y: FieldDescriptor[jnp.uint32]


# 1. Build the Stack
#    Stack.build(max_size, value_class)
stack = Stack.build(max_size=1000, value_class=Point)

# 2. Push a single item
p1 = Point(x=jnp.array(1, dtype=jnp.uint32), y=jnp.array(2, dtype=jnp.uint32))
stack = stack.push(p1)
print(f"Stack size after pushing one item: {stack.size}")

# 3. Push a batch of items
batch_points = Point(x=jnp.arange(10, dtype=jnp.uint32), y=jnp.arange(10, 20, dtype=jnp.uint32))
stack = stack.push(batch_points)
print(f"Stack size after pushing a batch: {stack.size}")

# 4. Peek at the top item
#    Does not modify the stack
peeked_item = stack.peek()
print("Peeked item:", peeked_item)
assert stack.size == 11  # Unchanged

# 5. Pop a batch of items
#    Removes the top 5 items from the stack
stack, popped_items = stack.pop(5)
print(f"Stack size after popping 5 items: {stack.size}")
print("Popped items (y-values):", popped_items.y)

# 6. Pop a single item
stack, popped_item = stack.pop()
print(f"Stack size after popping one item: {stack.size}")
print("Popped item:", popped_item)
```

## Key `Stack` Details

*   **LIFO Principle**: The last element added to the stack will be the first one to be removed.
*   **API Style**: The methods (`push`, `pop`) modify the stack's state and return the modified instance, allowing for a chained, functional-style usage pattern.

*   **`Stack.build(max_size, value_class)`**:
    *   `max_size` (int): The maximum number of elements the stack can hold.
    *   `value_class` (Xtructurable): The class of the data structure to be stored (e.g., `Point`). This defines the structure of the internal value store.

*   **`stack.push(items)`**:
    *   `items` (Xtructurable): An instance or a batch of instances to push onto the stack. If a batch is provided, its first dimension is treated as the batch dimension.
    *   Returns the updated `Stack` instance.

*   **`stack.pop(num_items=1)`**:
    *   `num_items` (int): The number of items to pop from the top of the stack.
    *   Returns a tuple containing:
        1.  The updated `Stack` instance with the items removed.
        2.  The `Xtructurable` containing the popped items.

*   **`stack.peek(num_items=1)`**:
    *   `num_items` (int): The number of items to view from the top of the stack.
    *   Returns the `Xtructurable` containing the top items without modifying the stack.

```
