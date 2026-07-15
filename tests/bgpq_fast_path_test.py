import os

import jax
import jax.numpy as jnp
import pytest

from tests.testdata import HeapValueABC
from xtructure import BGPQ

pytestmark = [
    pytest.mark.pallas_heap,
    pytest.mark.skipif(
        os.environ.get("XTRUCTURE_RUN_PALLAS_HEAP_TESTS") != "1",
        reason=(
            "BGPQ fast-path tests exercise the same hardware-specific JIT/Pallas lane "
            "as heap_test.py; set XTRUCTURE_RUN_PALLAS_HEAP_TESTS=1 to run them."
        ),
    ),
]


def test_empty_heap_insert_populates_root_without_buffering():
    batch_size = 4
    heap = BGPQ.build(16, batch_size, HeapValueABC, jnp.float32)
    keys = jnp.array([3.0, 1.0, jnp.inf, 2.0], dtype=jnp.float32)
    values = HeapValueABC.default((batch_size,))

    heap = heap.insert(keys, values)
    jax.block_until_ready(heap)

    assert heap.heap_size == 0
    assert heap.buffer_size == 0
    assert heap.size == 3
    assert jnp.all(heap.key_store[0] == jnp.array([1.0, 2.0, 3.0, jnp.inf]))


def test_sorted_insert_matches_regular_insert_for_sorted_batches():
    batch_size = 4
    regular = BGPQ.build(32, batch_size, HeapValueABC, jnp.float32)
    presorted = BGPQ.build(32, batch_size, HeapValueABC, jnp.float32)
    values = HeapValueABC.default((batch_size,)).replace(a=jnp.arange(batch_size, dtype=jnp.uint8))
    batches = (
        jnp.array([1.0, 4.0, 7.0, 10.0], dtype=jnp.float32),
        jnp.array([2.0, 3.0, 8.0, jnp.inf], dtype=jnp.float32),
        jnp.array([5.0, 5.0, 6.0, 9.0], dtype=jnp.float32),
    )

    for keys in batches:
        regular = regular.insert(keys, values)
        presorted = presorted.insert_sorted(keys, values)

    jax.block_until_ready((regular, presorted))

    regular_leaves = jax.tree_util.tree_leaves(regular)
    presorted_leaves = jax.tree_util.tree_leaves(presorted)
    assert len(regular_leaves) == len(presorted_leaves)
    assert all(jnp.array_equal(lhs, rhs) for lhs, rhs in zip(regular_leaves, presorted_leaves))


def test_empty_merge_buffer_stashes_partial_block_without_overflow():
    batch_size = 4
    heap = BGPQ.build(16, batch_size, HeapValueABC, jnp.float32)
    block_key = jnp.array([4.0, 5.0, jnp.inf, jnp.inf], dtype=jnp.float32)
    block_value = HeapValueABC.default((batch_size,))

    heap, out_key, _, overflow = heap.merge_buffer(block_key, block_value)
    jax.block_until_ready((heap, out_key, overflow))

    assert not overflow
    assert heap.buffer_size == 2
    assert jnp.all(jnp.isinf(out_key))
    assert jnp.all(heap.key_buffer == jnp.array([4.0, 5.0, jnp.inf]))


def test_empty_merge_buffer_overflows_full_block_without_reordering():
    batch_size = 4
    heap = BGPQ.build(16, batch_size, HeapValueABC, jnp.float32)
    block_key = jnp.array([4.0, 5.0, 6.0, 7.0], dtype=jnp.float32)
    block_value = HeapValueABC.default((batch_size,))

    heap, out_key, _, overflow = heap.merge_buffer(block_key, block_value)
    jax.block_until_ready((heap, out_key, overflow))

    assert overflow
    assert heap.buffer_size == 0
    assert jnp.all(out_key == block_key)
    assert jnp.all(jnp.isinf(heap.key_buffer))


def test_delete_single_root_without_buffer_clears_root():
    batch_size = 4
    heap = BGPQ.build(16, batch_size, HeapValueABC, jnp.float32)
    keys = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    values = HeapValueABC.default((batch_size,))
    heap = heap.insert(keys, values)

    heap, min_key, _ = BGPQ.delete_mins(heap)
    jax.block_until_ready((heap, min_key))

    assert heap.heap_size == 0
    assert heap.buffer_size == 0
    assert heap.size == 0
    assert jnp.all(min_key == keys)
    assert jnp.all(jnp.isinf(heap.key_store[0]))


def test_delete_single_root_fast_clear_allows_reinsert():
    batch_size = 4
    heap = BGPQ.build(16, batch_size, HeapValueABC, jnp.float32)
    values = HeapValueABC.default((batch_size,))
    heap = heap.insert(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32), values)
    heap, _, _ = BGPQ.delete_mins(heap)

    new_keys = jnp.array([7.0, 5.0, 8.0, 6.0], dtype=jnp.float32)
    heap = heap.insert(new_keys, values)
    jax.block_until_ready(heap)

    assert heap.heap_size == 0
    assert heap.buffer_size == 0
    assert heap.size == batch_size
    assert jnp.all(heap.key_store[0] == jnp.array([5.0, 6.0, 7.0, 8.0]))


def test_delete_single_root_refills_from_buffer_without_heap_nodes():
    batch_size = 4
    heap = BGPQ.build(16, batch_size, HeapValueABC, jnp.float32)
    root_keys = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    buffered_keys = jnp.array([5.0, 6.0, jnp.inf, jnp.inf], dtype=jnp.float32)
    values = HeapValueABC.default((batch_size,))
    heap = heap.insert(root_keys, values)
    heap = heap.insert(buffered_keys, values)

    heap, min_key, _ = BGPQ.delete_mins(heap)
    jax.block_until_ready((heap, min_key))

    assert heap.heap_size == 0
    assert heap.buffer_size == 0
    assert heap.size == 2
    assert jnp.all(min_key == root_keys)
    assert jnp.all(heap.key_store[0] == jnp.array([5.0, 6.0, jnp.inf, jnp.inf]))
    assert jnp.all(jnp.isinf(heap.key_buffer))


def test_delete_moves_last_node_to_root_when_buffer_is_empty():
    batch_size = 4
    heap = BGPQ.build(16, batch_size, HeapValueABC, jnp.float32)
    first_keys = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    second_keys = jnp.array([5.0, 6.0, 7.0, 8.0], dtype=jnp.float32)
    values = HeapValueABC.default((batch_size,))
    heap = heap.insert(first_keys, values)
    heap = heap.insert(second_keys, values)

    heap, min_key, _ = BGPQ.delete_mins(heap)
    jax.block_until_ready((heap, min_key))

    assert heap.heap_size == 0
    assert heap.buffer_size == 0
    assert heap.size == batch_size
    assert jnp.all(min_key == first_keys)
    assert jnp.all(heap.key_store[0] == second_keys)
    assert jnp.all(jnp.isinf(heap.key_store[1]))
