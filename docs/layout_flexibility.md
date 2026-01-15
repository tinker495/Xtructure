# Structure Layout Flexibility

Xtructure deliberately separates how data is stored from how you use it. Every
`@xtructure_dataclass` is backed by **Structure of Arrays (SoA)** tensors for
JAX performance, while the public API presents **Array of Structures (AoS)**
objects for clarity.

## Backend: SoA arrays for JAX

- Each field is declared with a `FieldDescriptor`, so the decorator stack can
  materialise a dedicated JAX array for the field when you call
  `default`, `random`, or any helper.
- Utilities such as `xnp.concat`, `HashTable.parallel_insert`, or
  `BGPQ.insert` work through `jax.tree_util.tree_map`, keeping all operations in
  the batched array world that JIT compilation, fusion, and vectorisation expect.

## Interface: AoS ergonomics for users

- Indexing (`state[0]`), `.at` updates, and container APIs (`Queue.dequeue`,
  `Stack.pop`) rewrap the mutated arrays into the original dataclass type, so you
  interact with plain Python objects.
- Nested dataclasses follow the same rule, allowing deeply structured states to
  feel idiomatic while preserving consistent SoA storage beneath.

## Bridging utilities

The `xtructure_numpy` module (`xnp`) provides layout-aware operations that work
seamlessly with `@xtructure_dataclass` instances:

```python
from xtructure import numpy as xnp  # Recommended import
```

- Helpers like `xnp.reshape`, `xnp.flatten`, `xnp.unique_mask`, and `xnp.pad`
  use tree maps to manipulate only batch axes, ensuring intrinsic field shapes
  remain intact.
- Hashing, serialisation, and deduplication reuse the SoA layout to derive byte
  representations or persistence formats without extra copying.

### Instance methods

The `@xtructure_dataclass` decorator also injects many `xnp` functions as
instance methods, so you can call them directly on dataclass instances:

```python
# These are equivalent:
reshaped = xnp.reshape(agents, (16, 8))
reshaped = agents.reshape((16, 8))

flipped = xnp.flip(agents, axis=0)
flipped = agents.flip(axis=0)
```

Available instance methods: `reshape`, `flatten`, `transpose`, `swapaxes`,
`moveaxis`, `squeeze`, `expand_dims`, `broadcast_to`, `roll`, `flip`, `rot90`,
`astype`, `pad`, `equal`, `not_equal`, `isclose`, `allclose`.

## Example

```python
import jax
import jax.numpy as jnp

from xtructure import FieldDescriptor, xtructure_dataclass


@xtructure_dataclass
class AgentState:
    pos: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))
    cost: FieldDescriptor.scalar(dtype=jnp.float32)


key = jax.random.PRNGKey(0)
agents = AgentState.random((128,), key=key)  # SoA storage for JIT speed
frontiers = agents.reshape((16, 8))  # reshape touches each field array
first = frontiers[0]  # AoS-style instance
updated = frontiers.at[0].set(first.replace(cost=jnp.zeros_like(first.cost)))
```

Behind the scenes this sequence performs field-wise JAX operations, yet each
step reads like ordinary dataclass manipulation.
