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

- Layout-aware helpers (`reshape`, `flatten`, `unique_mask`, `padding_as_batch`)
  use tree maps to manipulate only batch axes, ensuring intrinsic field shapes
  remain intact.
- Hashing, serialisation, and deduplication reuse the SoA layout to derive byte
  representations or persistence formats without extra copying.

## Example

```python
import jax
import jax.numpy as jnp

from xtructure import FieldDescriptor, xtructure_dataclass


@xtructure_dataclass
class AgentState:
    pos: FieldDescriptor[jnp.float32, (3,)]
    cost: FieldDescriptor[jnp.float32]


key = jax.random.PRNGKey(0)
agents = AgentState.random((128,), key=key)          # SoA storage for JIT speed
frontiers = agents.reshape((16, 8))                  # reshape touches each field array
first = frontiers[0]                                 # AoS-style instance
updated = frontiers.at[0].set(first.replace(cost=jnp.zeros_like(first.cost)))
```

Behind the scenes this sequence performs field-wise JAX operations, yet each
step reads like ordinary dataclass manipulation.
