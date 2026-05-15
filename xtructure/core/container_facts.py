"""Container Family shared facts.

The four xtructure containers (BGPQ / HashTable / Queue / Stack) share three
boilerplate facts that this Module concentrates as a single source of truth:

- ``SIZE_DTYPE``: the JAX dtype used for every container size / head / tail /
  heap_size counter. Containers that mix counters across families (e.g. a
  BGPQ holding HashIdx-shaped values) rely on a single dtype here so the same
  arithmetic survives JIT tracing.
- ``init_counter(value=0)``: a typed-zero constructor for the counter. Each
  container's ``build()`` factory used to redeclare ``jnp.uint32(0)`` inline;
  this helper concentrates that pattern.
- ``init_value_store(value_class, shape)``: the canonical
  ``value_class.default(shape)`` allocation. Each container varies the shape
  (Stack/Queue ask for ``(max_size,)``, HashTable adjusts for buckets, BGPQ
  uses ``(branch_size, batch_size)``), but they all need a single allocation
  helper so future changes to storage allocation policy land in one place.

Algorithm-specific hot paths (insert / pop / delete_mins / enqueue / dequeue
/ push) stay inside each container's own module. This Module owns only the
facts that every container must call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from .protocol import Xtructurable

SIZE_DTYPE = jnp.uint32


def init_counter(value: int = 0) -> jnp.uint32:
    """Return a ``SIZE_DTYPE``-typed counter initialised to ``value``.

    Used by every container's ``build()`` factory and any reset/clear path so
    the dtype matches across BGPQ / HashTable / Queue / Stack without each
    container redeclaring the literal cast.
    """
    return SIZE_DTYPE(value)


def init_value_store(value_class: type, shape: tuple[int, ...]) -> "Xtructurable":
    """Allocate an Xtructurable value store of ``shape``.

    Each container's ``build()`` factory uses ``value_class.default(shape)``
    to size its primary array; this helper concentrates that pattern so any
    future change to allocation policy (e.g. zero-padded vs. uninitialised,
    aligned vs. unaligned) happens once instead of four times.
    """
    return value_class.default(shape)
