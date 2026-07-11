from functools import partial

import jax
import jax.numpy as jnp

from ..core.dataclass import base_dataclass
from ..core.dtype_facts import SIZE_DTYPE
from ..core.packing import pack_rows, row_spec, unpack_rows
from ..core.protocol import Xtructurable


@partial(jax.jit, static_argnums=(0, 1))
def _queue_build_jit(max_size: int, value_class: Xtructurable):
    val_store = jnp.zeros((max_size, row_spec(value_class).row_bytes), dtype=jnp.uint8)
    head = SIZE_DTYPE(0)
    tail = SIZE_DTYPE(0)
    return Queue(
        max_size=max_size, value_class=value_class, val_store=val_store, head=head, tail=tail
    )


@jax.jit
def _queue_enqueue_jit(queue, items: Xtructurable):
    batch_size = items.shape.batch
    if batch_size == ():
        items = jax.tree_util.tree_map(lambda x: x[None], items)
        num_to_enqueue = 1
    else:
        assert len(batch_size) == 1, "Batch size must be 1"
        num_to_enqueue = batch_size[0]
    rows = pack_rows(queue.value_class, items)
    # One contiguous row write regardless of leaf count: the packed store
    # keeps the per-call GPU submission count flat (see core/packing.py).
    val_store = jax.lax.dynamic_update_slice(
        queue.val_store, rows, (queue.tail.astype(jnp.int32), jnp.int32(0))
    )
    new_tail = queue.tail + num_to_enqueue
    return queue.replace(val_store=val_store, tail=new_tail)


@partial(jax.jit, static_argnums=(1,))
def _queue_dequeue_jit(queue, num_items: int = 1):
    rows = jax.lax.dynamic_slice(
        queue.val_store,
        (queue.head.astype(jnp.int32), jnp.int32(0)),
        (num_items, queue.val_store.shape[1]),
    )
    items = unpack_rows(queue.value_class, rows)
    if num_items == 1:
        items = jax.tree_util.tree_map(lambda x: x[0], items)
    new_head = queue.head + num_items
    return queue.replace(head=new_head), items


@partial(jax.jit, static_argnums=(1,))
def _queue_peek_jit(queue, num_items: int = 1):
    rows = jax.lax.dynamic_slice(
        queue.val_store,
        (queue.head.astype(jnp.int32), jnp.int32(0)),
        (num_items, queue.val_store.shape[1]),
    )
    items = unpack_rows(queue.value_class, rows)
    if num_items == 1:
        items = jax.tree_util.tree_map(lambda x: x[0], items)
    return items


@jax.jit
def _queue_clear_jit(queue):
    return queue.replace(head=SIZE_DTYPE(0), tail=SIZE_DTYPE(0))


@jax.jit
def _queue_getitem_jit(queue, idx: SIZE_DTYPE) -> Xtructurable:
    storage_idx = (queue.head + idx).astype(jnp.int32)
    rows = jax.lax.dynamic_slice(
        queue.val_store, (storage_idx, jnp.int32(0)), (1, queue.val_store.shape[1])
    )
    return jax.tree_util.tree_map(lambda x: x[0], unpack_rows(queue.value_class, rows))


@base_dataclass(static_fields=("max_size", "value_class"))
class Queue:
    """
    A JAX-compatible batched Queue data structure.
    Optimized for parallel operations on GPU using JAX.

    Attributes:
        max_size: Maximum number of elements the queue can hold.
        value_class: The Xtructurable class stored in the queue.
        val_store: Packed byte storage, ``uint8[max_size, row_bytes]``.
        head: Index of the first item in the queue.
        tail: Index of the next available slot.
    """

    max_size: int
    value_class: Xtructurable
    val_store: jnp.ndarray
    head: SIZE_DTYPE
    tail: SIZE_DTYPE

    @property
    def size(self):
        return self.tail - self.head

    @staticmethod
    def build(max_size: int, value_class: Xtructurable) -> "Queue":
        """
        Creates a new Queue instance.
        """
        return _queue_build_jit(max_size, value_class)

    def enqueue(self, items: Xtructurable) -> "Queue":
        """
        Enqueues a number of items into the queue.
        """
        return _queue_enqueue_jit(self, items)

    def dequeue(self, num_items: int = 1) -> tuple["Queue", Xtructurable]:
        """
        Dequeues a number of items from the queue.
        """
        return _queue_dequeue_jit(self, num_items)

    def peek(self, num_items: int = 1) -> Xtructurable:
        """
        Peeks at the front items of the queue without removing them.
        """
        return _queue_peek_jit(self, num_items)

    def clear(self) -> "Queue":
        """
        Clears the queue.
        """
        return _queue_clear_jit(self)

    def __getitem__(self, idx: SIZE_DTYPE) -> Xtructurable:
        """
        Returns the item at the logical queue index (0-based, relative to head).
        """
        return _queue_getitem_jit(self, idx)
