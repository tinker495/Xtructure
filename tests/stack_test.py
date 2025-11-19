import chex
import jax
import jax.numpy as jnp
import pytest

from xtructure import FieldDescriptor, Stack, xtructure_dataclass


@xtructure_dataclass
class Point:
    x: FieldDescriptor.scalar(dtype=jnp.uint32)
    y: FieldDescriptor.scalar(dtype=jnp.uint32)


# Use a much larger max_size for more robust testing
LARGE_MAX_SIZE = 100_000


@pytest.fixture
def stack():
    """Provides a fresh stack for each test."""
    return Stack.build(max_size=LARGE_MAX_SIZE, value_class=Point)


def test_build(stack):
    """Tests the initial state of a newly built stack."""
    assert stack.size == 0
    assert stack.max_size == LARGE_MAX_SIZE


def test_push_single_item(stack):
    """Tests pushing a single item onto the stack."""
    p1 = Point(x=jnp.array(1, dtype=jnp.uint32), y=jnp.array(2, dtype=jnp.uint32))

    stack = stack.push(p1)

    assert stack.size == 1
    peeked = stack.peek()
    # peek returns a batch of 1
    chex.assert_trees_all_equal(peeked, p1)


def test_push_batch(stack):
    """Tests pushing a batch of items onto the stack."""
    batch_size = 5000
    points = Point(
        x=jnp.arange(batch_size, dtype=jnp.uint32),
        y=jnp.arange(batch_size, batch_size * 2, dtype=jnp.uint32),
    )

    stack = stack.push(points)

    assert stack.size == batch_size
    peeked = stack.peek(batch_size)
    chex.assert_trees_all_equal(peeked, points)


def test_pop_single(stack):
    """Tests popping items one by one."""
    p1 = Point(x=jnp.array(1, dtype=jnp.uint32), y=jnp.array(2, dtype=jnp.uint32))
    p2 = Point(x=jnp.array(3, dtype=jnp.uint32), y=jnp.array(4, dtype=jnp.uint32))

    stack = stack.push(p1)
    stack = stack.push(p2)

    assert stack.size == 2

    stack, popped = stack.pop()
    assert stack.size == 1
    # pop returns a batch of 1
    chex.assert_trees_all_equal(popped, p2)

    stack, popped = stack.pop()
    assert stack.size == 0
    chex.assert_trees_all_equal(popped, p1)


def test_pop_batch(stack):
    """Tests popping a batch of items."""
    batch_size = 5000
    points = Point(
        x=jnp.arange(batch_size, dtype=jnp.uint32),
        y=jnp.arange(batch_size, batch_size * 2, dtype=jnp.uint32),
    )
    stack = stack.push(points)

    pop_count = 3000
    stack, popped = stack.pop(pop_count)

    assert stack.size == batch_size - pop_count
    chex.assert_trees_all_equal(
        popped,
        Point(
            x=jnp.arange(batch_size - pop_count, batch_size, dtype=jnp.uint32),
            y=jnp.arange(batch_size * 2 - pop_count, batch_size * 2, dtype=jnp.uint32),
        ),
    )


def test_peek(stack):
    """Tests peeking without modifying the stack."""
    batch_size = 5000
    points = Point(
        x=jnp.arange(batch_size, dtype=jnp.uint32),
        y=jnp.arange(batch_size, batch_size * 2, dtype=jnp.uint32),
    )
    stack = stack.push(points)

    original_size = stack.size
    peek_count = 3000
    peeked = stack.peek(peek_count)

    assert stack.size == original_size
    chex.assert_trees_all_equal(
        peeked,
        Point(
            x=jnp.arange(batch_size - peek_count, batch_size, dtype=jnp.uint32),
            y=jnp.arange(batch_size * 2 - peek_count, batch_size * 2, dtype=jnp.uint32),
        ),
    )


def test_jit_compatibility(stack):
    @jax.jit
    def sequence(stack):
        p1 = Point(x=jnp.array(1, dtype=jnp.uint32), y=jnp.array(2, dtype=jnp.uint32))
        p2 = Point(x=jnp.arange(2, dtype=jnp.uint32), y=jnp.arange(2, 4, dtype=jnp.uint32))

        stack = stack.push(p1)
        stack = stack.push(p2)
        stack, _ = stack.pop(2)
        return stack

    final_stack = sequence(stack)
    assert final_stack.size == 1
