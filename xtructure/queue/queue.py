from functools import partial

import jax
import jax.numpy as jnp

from ..core import Xtructurable, base_dataclass

SIZE_DTYPE = jnp.uint32


@partial(jax.jit, static_argnums=(0, 1))
def _queue_build_jit(max_size: int, value_class: Xtructurable):
    val_store = value_class.default((max_size,))
    head = SIZE_DTYPE(0)
    tail = SIZE_DTYPE(0)
    return Queue(max_size=max_size, val_store=val_store, head=head, tail=tail)


@jax.jit
def _queue_enqueue_jit(queue, items: Xtructurable):
    batch_size = items.shape.batch
    if batch_size == ():
        num_to_enqueue = 1
        indices = queue.tail
    else:
        assert len(batch_size) == 1, "Batch size must be 1"
        num_to_enqueue = batch_size[0]
        indices = queue.tail + jnp.arange(num_to_enqueue)
    val_store = queue.val_store.at[indices].set(items)
    new_tail = queue.tail + num_to_enqueue
    return queue.replace(val_store=val_store, tail=new_tail)


@partial(jax.jit, static_argnums=(1,))
def _queue_dequeue_jit(queue, num_items: int = 1):
    if num_items == 1:
        indices = queue.head
    else:
        indices = queue.head + jnp.arange(num_items)

    dequeued_items = queue.val_store[indices]
    new_head = queue.head + num_items
    return queue.replace(head=new_head), dequeued_items


@partial(jax.jit, static_argnums=(1,))
def _queue_peek_jit(queue, num_items: int = 1):
    if num_items == 1:
        indices = queue.head
    else:
        indices = queue.head + jnp.arange(num_items)
    peeked_items = queue.val_store[indices]
    return peeked_items


@jax.jit
def _queue_clear_jit(queue):
    return queue.replace(head=SIZE_DTYPE(0), tail=SIZE_DTYPE(0))


@jax.jit
def _queue_getitem_jit(queue, idx: SIZE_DTYPE) -> Xtructurable:
    storage_idx = queue.head + idx
    return queue.val_store[storage_idx]


@base_dataclass(static_fields=("max_size",))
class Queue:
    """
    A JAX-compatible batched Queue data structure.
    Optimized for parallel operations on GPU using JAX.

    Attributes:
        max_size: Maximum number of elements the queue can hold.
        val_store: Array storing the values in the queue.
        head: Index of the first item in the queue.
        tail: Index of the next available slot.
    """

    max_size: int
    val_store: Xtructurable
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
