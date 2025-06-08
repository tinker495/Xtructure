from functools import partial

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable

SIZE_DTYPE = jnp.uint32


@chex.dataclass
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
    @partial(jax.jit, static_argnums=(0, 1))
    def build(max_size: int, value_class: Xtructurable):
        """
        Creates a new Queue instance.
        """
        val_store = value_class.default((max_size,))
        head = SIZE_DTYPE(0)
        tail = SIZE_DTYPE(0)
        return Queue(max_size=max_size, val_store=val_store, head=head, tail=tail)

    @jax.jit
    def enqueue(self, items: Xtructurable):
        """
        Enqueues a number of items into the queue.
        """
        batch_size = items.shape.batch
        if batch_size == ():
            num_to_enqueue = 1
            indices = self.tail
        else:
            assert len(batch_size) == 1, "Batch size must be 1"
            num_to_enqueue = batch_size[0]
            indices = self.tail + jnp.arange(num_to_enqueue)
        self.val_store = self.val_store.at[indices].set(items)
        self.tail = self.tail + num_to_enqueue
        return self

    @partial(jax.jit, static_argnums=(1,))
    def dequeue(self, num_items: int = 1):
        """
        Dequeues a number of items from the queue.
        """
        if num_items == 1:
            indices = self.head
        else:
            indices = self.head + jnp.arange(num_items)

        dequeued_items = self.val_store[indices]
        self.head = self.head + num_items
        return self, dequeued_items

    @partial(jax.jit, static_argnums=(1,))
    def peek(self, num_items: int = 1):
        """
        Peeks at the front items of the queue without removing them.
        """
        if num_items == 1:
            indices = self.head
        else:
            indices = self.head + jnp.arange(num_items)
        peeked_items = self.val_store[indices]
        return peeked_items

    @jax.jit
    def clear(self):
        """
        Clears the queue.
        """
        self.head = SIZE_DTYPE(0)
        self.tail = SIZE_DTYPE(0)
        return self
