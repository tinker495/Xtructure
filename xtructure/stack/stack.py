from functools import partial

import jax
import jax.numpy as jnp

from ..core import Xtructurable, base_dataclass

SIZE_DTYPE = jnp.uint32


@partial(jax.jit, static_argnums=(0, 1))
def _stack_build_jit(max_size: int, value_class: Xtructurable):
    size = SIZE_DTYPE(0)
    val_store = value_class.default((max_size,))
    return Stack(max_size=max_size, size=size, val_store=val_store)


@jax.jit
def _stack_push_jit(stack, items: Xtructurable):
    batch_size = items.shape.batch
    if batch_size == ():
        new_size = stack.size + 1
        indices = stack.size
    else:
        assert len(batch_size) == 1, "Batch size must be 1"
        new_size = stack.size + batch_size[0]
        indices = stack.size + jnp.arange(batch_size[0])
    val_store = stack.val_store.at[indices].set(items)
    # Since Stack is a dataclass (chex.ArrayTree), we need to return a new instance
    # or if it's mutable (which it isn't usually in JAX), we construct a new one.
    # The original code was modifying self attributes which is not pure JAX if it was a python class
    # but here it's @base_dataclass which is likely a Pytree.
    # The original code did: self.val_store = ...; return self.
    # We should reconstruct.
    return stack.replace(val_store=val_store, size=new_size)


@partial(jax.jit, static_argnums=(1,))
def _stack_pop_jit(stack, num_items: int = 1):
    new_size = stack.size - num_items
    if num_items == 1:
        indices = stack.size - 1
    else:
        indices = stack.size - jnp.arange(num_items, 0, -1)
    popped_items = stack.val_store[indices]
    return stack.replace(size=new_size), popped_items


@partial(jax.jit, static_argnums=(1,))
def _stack_peek_jit(stack, num_items: int = 1):
    if num_items == 1:
        indices = stack.size - 1
    else:
        indices = stack.size - jnp.arange(num_items, 0, -1)
    peeked_items = stack.val_store[indices]
    return peeked_items


@jax.jit
def _stack_getitem_jit(stack, idx: SIZE_DTYPE) -> Xtructurable:
    return stack.val_store[idx]


@base_dataclass(static_fields=("max_size",))
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
    def build(max_size: int, value_class: Xtructurable) -> "Stack":
        """
        Creates a new Stack instance.

        Args:
            max_size: The maximum number of elements the stack can hold.
            value_class: The class of values to be stored in the stack.
                        It must be a subclass of Xtructurable.

        Returns:
            A new, empty Stack instance.
        """
        return _stack_build_jit(max_size, value_class)

    def push(self, items: Xtructurable) -> "Stack":
        """
        Pushes a batch of items onto the stack.

        Args:
            items: An Xtructurable containing the items to push. The first
                   dimension is the batch dimension.

        Returns:
            A new Stack instance with the items pushed onto it.
        """
        return _stack_push_jit(self, items)

    def pop(self, num_items: int = 1) -> tuple["Stack", Xtructurable]:
        """
        Pops a number of items from the stack.

        Args:
            num_items: The number of items to pop.

        Returns:
            A tuple containing:
                - A new Stack instance with items removed.
                - The popped items.
        """
        return _stack_pop_jit(self, num_items)

    def peek(self, num_items: int = 1) -> Xtructurable:
        """
        Peeks at the top items of the stack without removing them.

        Args:
            num_items: The number of items to peek at. Defaults to 1.

        Returns:
            The top `num_items` from the stack.
        """
        return _stack_peek_jit(self, num_items)

    def __getitem__(self, idx: SIZE_DTYPE) -> Xtructurable:
        """
        Returns the item at the logical stack index (0-based, relative to bottom).
        """
        return _stack_getitem_jit(self, idx)
