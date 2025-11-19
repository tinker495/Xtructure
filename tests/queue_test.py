import chex
import jax
import jax.numpy as jnp
import pytest

from xtructure import FieldDescriptor, Queue, xtructure_dataclass


@xtructure_dataclass
class Point:
    x: FieldDescriptor.scalar(dtype=jnp.uint32)
    y: FieldDescriptor.scalar(dtype=jnp.uint32)


LARGE_MAX_SIZE = 100_000


@pytest.fixture
def queue():
    """Provides a fresh queue for each test."""
    return Queue.build(max_size=LARGE_MAX_SIZE, value_class=Point)


def test_build(queue):
    """Tests the initial state of a newly built queue."""
    assert queue.size == 0
    assert queue.max_size == LARGE_MAX_SIZE
    assert queue.head == 0
    assert queue.tail == 0


def test_enqueue_single_item(queue):
    """Tests enqueuing a single item."""
    p1 = Point(x=jnp.array(1, dtype=jnp.uint32), y=jnp.array(2, dtype=jnp.uint32))
    queue = queue.enqueue(p1)
    assert queue.size == 1
    assert queue.tail == 1
    peeked = queue.peek()
    chex.assert_trees_all_equal(peeked, p1)


def test_enqueue_batch(queue):
    """Tests enqueuing a batch of items."""
    batch_size = 5000
    points = Point(
        x=jnp.arange(batch_size, dtype=jnp.uint32),
        y=jnp.arange(batch_size, batch_size * 2, dtype=jnp.uint32),
    )
    queue = queue.enqueue(points)
    assert queue.size == batch_size
    assert queue.tail == batch_size
    peeked = queue.peek(batch_size)
    chex.assert_trees_all_equal(peeked, points)


def test_dequeue_single(queue):
    """Tests dequeuing items one by one."""
    p1 = Point(x=jnp.array(1, dtype=jnp.uint32), y=jnp.array(2, dtype=jnp.uint32))
    p2 = Point(x=jnp.array(3, dtype=jnp.uint32), y=jnp.array(4, dtype=jnp.uint32))

    queue = queue.enqueue(p1)
    queue = queue.enqueue(p2)
    assert queue.size == 2

    queue, dequeued = queue.dequeue()
    assert queue.size == 1
    assert queue.head == 1
    chex.assert_trees_all_equal(dequeued, p1)

    queue, dequeued = queue.dequeue()
    assert queue.size == 0
    assert queue.head == 2
    chex.assert_trees_all_equal(dequeued, p2)


def test_dequeue_batch(queue):
    """Tests dequeuing a batch of items."""
    batch_size = 5000
    points = Point(
        x=jnp.arange(batch_size, dtype=jnp.uint32),
        y=jnp.arange(batch_size, batch_size * 2, dtype=jnp.uint32),
    )
    queue = queue.enqueue(points)

    dequeue_count = 3000
    queue, dequeued = queue.dequeue(dequeue_count)

    assert queue.size == batch_size - dequeue_count
    assert queue.head == dequeue_count
    expected_dequeued = Point(x=points.x[:dequeue_count], y=points.y[:dequeue_count])
    chex.assert_trees_all_equal(dequeued, expected_dequeued)


def test_peek(queue):
    """Tests peeking without modifying the queue."""
    batch_size = 5000
    points = Point(
        x=jnp.arange(batch_size, dtype=jnp.uint32),
        y=jnp.arange(batch_size, batch_size * 2, dtype=jnp.uint32),
    )
    queue = queue.enqueue(points)

    original_size = queue.size
    original_head = queue.head
    peek_count = 3000
    peeked = queue.peek(peek_count)

    assert queue.size == original_size
    assert queue.head == original_head

    expected_peeked = Point(x=points.x[:peek_count], y=points.y[:peek_count])
    chex.assert_trees_all_equal(peeked, expected_peeked)


def test_clear(queue):
    """Tests clearing the queue."""
    points = Point(x=jnp.arange(5, dtype=jnp.uint32), y=jnp.arange(5, 10, dtype=jnp.uint32))
    queue = queue.enqueue(points)
    assert queue.size == 5

    queue = queue.clear()
    assert queue.size == 0
    assert queue.head == 0
    assert queue.tail == 0


def test_jit_compatibility(queue):
    @jax.jit
    def sequence(q):
        p1 = Point(x=jnp.array(1, dtype=jnp.uint32), y=jnp.array(2, dtype=jnp.uint32))
        batch_points = Point(
            x=jnp.arange(2, dtype=jnp.uint32), y=jnp.arange(2, 4, dtype=jnp.uint32)
        )

        q = q.enqueue(p1)
        q = q.enqueue(batch_points)
        q, _ = q.dequeue(2)
        return q

    final_queue = sequence(queue)
    assert final_queue.size == 1
    assert final_queue.head == 2
    assert final_queue.tail == 3
