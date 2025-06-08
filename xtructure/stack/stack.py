from functools import partial

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable

SIZE_DTYPE = jnp.uint32


@chex.dataclass
class Stack:
    """
    A JAX-compatible batched Stack data structure.
    Optimized for parallel operations on GPU using JAX.

    Attributes:
        max_size: Maximum number of elements the stack can hold.
        size: Current number of elements in the stack.
        val_store: Array storing the values in the stack.
    """

    max_size: int
    size: SIZE_DTYPE
    val_store: Xtructurable

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def build(max_size: int, value_class: Xtructurable):
        """
        Creates a new Stack instance.

        Args:
            max_size: The maximum number of elements the stack can hold.
            value_class: The class of values to be stored in the stack.
                        It must be a subclass of Xtructurable.

        Returns:
            A new, empty Stack instance.
        """
        size = SIZE_DTYPE(0)
        val_store = value_class.default((max_size,))
        return Stack(max_size=max_size, size=size, val_store=val_store)

    @jax.jit
    def push(self, items: Xtructurable):
        """
        Pushes a batch of items onto the stack.

        Args:
            items: An Xtructurable containing the items to push. The first
                   dimension is the batch dimension.

        Returns:
            A new Stack instance with the items pushed onto it.
        """
        batch_size = items.shape.batch
        if batch_size == ():
            new_size = self.size + 1
            indices = self.size
        else:
            assert len(batch_size) == 1, "Batch size must be 1"
            new_size = self.size + batch_size[0]
            indices = self.size + jnp.arange(batch_size[0])
        self.val_store = self.val_store.at[indices].set(items)
        self.size = new_size
        return self

    @partial(jax.jit, static_argnums=(1,))
    def pop(self, num_items: int = 1):
        """
        Pops a number of items from the stack.

        Args:
            num_items: The number of items to pop.

        Returns:
            A tuple containing:
                - A new Stack instance with items removed.
                - The popped items.
        """
        new_size = self.size - num_items
        if num_items == 1:
            indices = self.size - 1
        else:
            indices = self.size - jnp.arange(num_items, 0, -1)
        popped_items = self.val_store[indices]
        self.size = new_size
        return self, popped_items

    @partial(jax.jit, static_argnums=(1,))
    def peek(self, num_items: int = 1):
        """
        Peeks at the top items of the stack without removing them.

        Args:
            num_items: The number of items to peek at. Defaults to 1.

        Returns:
            The top `num_items` from the stack.
        """
        if num_items == 1:
            indices = self.size - 1
        else:
            indices = self.size - jnp.arange(num_items, 0, -1)
        peeked_items = self.val_store[indices]
        return peeked_items
