import jax
import jax.numpy as jnp

from xtructure.bgpq.merge_split.common import _get_sentinels, binary_search_partition


def test_get_sentinels_float():
    min_val, max_val = _get_sentinels(jnp.dtype(jnp.float32))
    assert min_val == jnp.finfo(jnp.float32).min
    assert max_val == jnp.finfo(jnp.float32).max


def test_get_sentinels_int():
    min_val, max_val = _get_sentinels(jnp.dtype(jnp.int32))
    assert min_val == jnp.iinfo(jnp.int32).min
    assert max_val == jnp.iinfo(jnp.int32).max


def test_binary_search_partition():
    a = jnp.array([1, 3, 5, 7, 9], dtype=jnp.int32)
    b = jnp.array([2, 4, 6, 8, 10], dtype=jnp.int32)

    i, j = jax.jit(binary_search_partition)(4, a, b)
    assert i == 2
    assert j == 2

    i, j = jax.jit(binary_search_partition)(1, a, b)
    assert i == 1
    assert j == 0

    i, j = jax.jit(binary_search_partition)(10, a, b)
    assert i == 5
    assert j == 5

    i, j = jax.jit(binary_search_partition)(0, a, b)
    assert i == 0
    assert j == 0
