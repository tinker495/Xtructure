from functools import partial

import jax
import jax.numpy as jnp

from ..core.dataclass import base_dataclass
from ..core.dtype_facts import SIZE_DTYPE
from ..core.packing import pack_rows, unpack_rows
from ..core.protocol import Xtructurable


@partial(jax.jit, static_argnums=(0, 1))
def _stack_build_jit(max_size: int, value_class: Xtructurable):
    size = SIZE_DTYPE(0)
    # Packed default rows, not zeros: reads of never-written slots (e.g.
    # clamped partial-pop rows) must keep returning value_class defaults.
    val_store = pack_rows(value_class, value_class.default((max_size,)))
    return Stack(max_size=max_size, value_class=value_class, size=size, val_store=val_store)


@jax.jit
def _stack_push_jit(stack, items: Xtructurable):
    batch_size = items.shape.batch
    if batch_size == ():
        items = jax.tree_util.tree_map(lambda x: x[None], items)
        num_to_push = 1
    else:
        assert len(batch_size) == 1, "Batch size must be 1"
        num_to_push = batch_size[0]
    rows = pack_rows(stack.value_class, items)
    # One contiguous row write regardless of leaf count: the packed store
    # keeps the per-call GPU submission count flat (see core/packing.py).
    val_store = jax.lax.dynamic_update_slice(
        stack.val_store, rows, (stack.size.astype(jnp.int32), jnp.int32(0))
    )
    return stack.replace(val_store=val_store, size=stack.size + num_to_push)


@partial(jax.jit, static_argnums=(1,))
def _stack_read_jit(stack, num_items: int):
    # Gather with the legacy index arithmetic (uint32 underflow + clamp on
    # partial pops) — JAxtar id_stars depends on that exact row placement,
    # so a start-clamping dynamic_slice is NOT equivalent here.
    if num_items == 1:
        indices = (stack.size - 1)[None]
    else:
        indices = stack.size - jnp.arange(num_items, 0, -1)
    rows = stack.val_store[indices]
    items = unpack_rows(stack.value_class, rows)
    if num_items == 1:
        items = jax.tree_util.tree_map(lambda x: x[0], items)
    return items


@partial(jax.jit, static_argnums=(1,))
def _stack_pop_jit(stack, num_items: int = 1):
    popped_items = _stack_read_jit(stack, num_items)
    return stack.replace(size=stack.size - num_items), popped_items


@jax.jit
def _stack_getitem_jit(stack, idx: SIZE_DTYPE) -> Xtructurable:
    rows = stack.val_store[jnp.asarray(idx)[None]]
    return jax.tree_util.tree_map(lambda x: x[0], unpack_rows(stack.value_class, rows))


@base_dataclass(static_fields=("max_size", "value_class"))
class Stack:
    """
    A JAX-compatible batched Stack data structure.
    Optimized for parallel operations on GPU using JAX.

    Attributes:
        max_size: Maximum number of elements the stack can hold.
        value_class: The Xtructurable class stored in the stack.
        size: Current number of elements in the stack.
        val_store: Packed byte storage, ``uint8[max_size, row_bytes]``.
    """

    max_size: int
    value_class: Xtructurable
    size: SIZE_DTYPE
    val_store: jnp.ndarray

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
        return _stack_read_jit(self, num_items)

    def __getitem__(self, idx: SIZE_DTYPE) -> Xtructurable:
        """
        Returns the item at the logical stack index (0-based, relative to bottom).
        """
        return _stack_getitem_jit(self, idx)
