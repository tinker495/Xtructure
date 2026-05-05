"""Tests for newer helper operations like expand_dims, repeat, split, and zeros_like."""

import jax.numpy as jnp
import pytest

from tests.xnp.shared_data import SimpleData, VectorData
from xtructure import numpy as xnp
from xtructure.core import xtructure_numpy as xnp_impl


def test_expand_dims_single_dataclass():
    """Expand dims should add a new leading axis."""
    data = SimpleData.default()
    data = data.replace(id=jnp.array(7, dtype=jnp.uint32), value=jnp.array(3.14, dtype=jnp.float32))

    expanded = xnp.expand_dims(data, axis=0)

    assert expanded.structured_type.name == "BATCHED"
    assert expanded.shape.batch == (1,)
    assert jnp.array_equal(expanded.id, jnp.array([7], dtype=jnp.uint32))
    assert jnp.array_equal(expanded.value, jnp.array([3.14], dtype=jnp.float32))


def test_expand_dims_batched_axis():
    """Expand dims on batched dataclass should insert axis at requested position."""
    data = SimpleData.default(shape=(2,))
    data = data.replace(id=jnp.array([1, 2], dtype=jnp.uint32), value=jnp.array([1.0, 2.0]))

    expanded = xnp.expand_dims(data, axis=1)

    assert expanded.shape.batch == (2, 1)
    assert jnp.array_equal(expanded.id, data.id[:, None])
    assert jnp.array_equal(expanded.value, data.value[:, None])


def test_squeeze_removes_unit_axes():
    """Squeeze should remove axes of size 1."""
    data = SimpleData.default(shape=(1, 3, 1))
    data = data.replace(
        id=jnp.arange(3, dtype=jnp.uint32).reshape(1, 3, 1),
        value=jnp.arange(3, dtype=jnp.float32).reshape(1, 3, 1),
    )

    squeezed = xnp.squeeze(data)

    assert squeezed.shape.batch == (3,)
    assert jnp.array_equal(squeezed.id, jnp.arange(3, dtype=jnp.uint32))
    assert jnp.array_equal(squeezed.value, jnp.arange(3, dtype=jnp.float32))


def test_repeat_matches_jnp_repeat():
    """Repeat should match jnp.repeat behaviour on each field."""
    data = SimpleData.default(shape=(2,))
    data = data.replace(
        id=jnp.array([1, 2], dtype=jnp.uint32),
        value=jnp.array([10.0, 20.0], dtype=jnp.float32),
    )

    repeated = xnp.repeat(data, repeats=2, axis=0)

    expected_id = jnp.repeat(data.id, repeats=2, axis=0)
    expected_value = jnp.repeat(data.value, repeats=2, axis=0)
    assert repeated.shape.batch == (4,)
    assert jnp.array_equal(repeated.id, expected_id)
    assert jnp.array_equal(repeated.value, expected_value)


def test_repeat_no_axis_defaults_flattened():
    """Repeat without axis should behave like jnp.repeat on flattened entries."""
    data = SimpleData.default(shape=(2, 2))
    data = data.replace(
        id=jnp.array([[1, 2], [3, 4]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
    )

    repeated = xnp.repeat(data, repeats=2)
    expected_id = jnp.repeat(data.id, repeats=2)
    expected_value = jnp.repeat(data.value, repeats=2)
    assert jnp.array_equal(repeated.id, expected_id)
    assert jnp.array_equal(repeated.value, expected_value)


def test_split_even_sections():
    """Split should divide the dataclass into even sections."""
    data = SimpleData.default(shape=(4,))
    data = data.replace(
        id=jnp.array([1, 2, 3, 4], dtype=jnp.uint32),
        value=jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32),
    )

    parts = xnp.split(data, 2)

    assert len(parts) == 2
    assert all(part.shape.batch == (2,) for part in parts)
    assert jnp.array_equal(parts[0].id, jnp.array([1, 2], dtype=jnp.uint32))
    assert jnp.array_equal(parts[1].value, jnp.array([30.0, 40.0], dtype=jnp.float32))


def test_split_by_indices():
    """Split with explicit indices should respect split points."""
    data = SimpleData.default(shape=(5,))
    data = data.replace(id=jnp.arange(5, dtype=jnp.uint32), value=jnp.arange(5, dtype=jnp.float32))

    parts = xnp.split(data, jnp.array([2, 4]))

    assert len(parts) == 3
    assert parts[0].shape.batch == (2,)
    assert parts[1].shape.batch == (2,)
    assert parts[2].shape.batch == (1,)
    assert jnp.array_equal(parts[2].id, jnp.array([4], dtype=jnp.uint32))


def test_zeros_ones_full_like_helpers():
    """zeros_like / ones_like / full_like should mirror jnp helpers."""
    data = VectorData.default(shape=(2,))
    data = data.replace(
        position=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        velocity=jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    )

    zeros = xnp.zeros_like(data)
    ones = xnp.ones_like(data)
    full = xnp.full_like(data, 7.5)

    assert jnp.array_equal(zeros.position, jnp.zeros_like(data.position))
    assert jnp.array_equal(ones.velocity, jnp.ones_like(data.velocity))
    assert jnp.array_equal(full.position, jnp.full_like(data.position, 7.5))
    assert zeros.position.dtype == data.position.dtype
    assert ones.velocity.dtype == data.velocity.dtype


def test_zeros_ones_like_do_not_forward_default_out_sharding(monkeypatch):
    """Default out_sharding should not be sent to older JAX *_like callables."""
    original_zeros_like = xnp_impl.jnp.zeros_like
    original_ones_like = xnp_impl.jnp.ones_like

    def legacy_zeros_like(a, dtype=None, shape=None, *, device=None):
        return original_zeros_like(a, dtype=dtype, shape=shape, device=device)

    def legacy_ones_like(a, dtype=None, shape=None, *, device=None):
        return original_ones_like(a, dtype=dtype, shape=shape, device=device)

    monkeypatch.setattr(xnp_impl.jnp, "zeros_like", legacy_zeros_like)
    monkeypatch.setattr(xnp_impl.jnp, "ones_like", legacy_ones_like)

    arr = jnp.arange(3, dtype=jnp.float32)
    assert jnp.array_equal(xnp.zeros_like(arr), original_zeros_like(arr))
    assert jnp.array_equal(xnp.ones_like(arr), original_ones_like(arr))

    with pytest.raises(TypeError, match="out_sharding requires"):
        xnp.zeros_like(arr, out_sharding=object())
