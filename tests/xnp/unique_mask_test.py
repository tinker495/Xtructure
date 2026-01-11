"""Tests for xnp.unique_mask helper."""

import jax.numpy as jnp
import pytest

from tests.xnp.shared_data import HashableData
from xtructure import numpy as xnp


def test_unique_mask_basic_uniqueness():
    """Test basic uniqueness without cost consideration."""
    data = HashableData.default(shape=(5,))
    data = data.replace(id=jnp.array([1, 2, 1, 3, 2]), value=jnp.array([1.0, 2.0, 1.0, 3.0, 2.0]))

    mask = xnp.unique_mask(data)

    expected_mask = jnp.array([True, True, False, True, False])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_with_costs():
    """Test unique filtering with cost-based selection."""
    data = HashableData.default(shape=(6,))
    data = data.replace(
        id=jnp.array([1, 2, 1, 3, 2, 1]),
        value=jnp.array([1.0, 2.0, 1.0, 3.0, 2.0, 1.0]),
    )

    costs = jnp.array([5.0, 3.0, 2.0, 4.0, 7.0, 1.0])

    mask = xnp.unique_mask(data, key=costs)

    expected_mask = jnp.array([False, True, False, True, False, True])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_tie_breaking():
    """Test tie-breaking when costs are equal."""
    data = HashableData.default(shape=(4,))
    data = data.replace(id=jnp.array([1, 2, 1, 2]), value=jnp.array([1.0, 2.0, 1.0, 2.0]))
    costs = jnp.array([3.0, 4.0, 3.0, 4.0])

    mask = xnp.unique_mask(data, key=costs)

    expected_mask = jnp.array([True, True, False, False])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_infinite_costs():
    """Test that entries with infinite cost are excluded."""
    data = HashableData.default(shape=(4,))
    data = data.replace(id=jnp.array([1, 2, 1, 2]), value=jnp.array([1.0, 2.0, 1.0, 2.0]))
    costs = jnp.array([1.0, jnp.inf, 2.0, 3.0])

    mask = xnp.unique_mask(data, key=costs)

    expected_mask = jnp.array([True, False, False, True])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_all_same():
    """Test when all elements are the same."""
    data = HashableData.default(shape=(3,))
    data = data.replace(id=jnp.array([1, 1, 1]), value=jnp.array([1.0, 1.0, 1.0]))
    costs = jnp.array([3.0, 1.0, 2.0])

    mask = xnp.unique_mask(data, key=costs)

    expected_mask = jnp.array([False, True, False])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_all_unique():
    """Test when all elements are unique."""
    data = HashableData.default(shape=(3,))
    data = data.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))
    costs = jnp.array([3.0, 1.0, 2.0])

    mask = xnp.unique_mask(data, key=costs)

    expected_mask = jnp.array([True, True, True])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_explicit_batch_len():
    """Test with explicitly provided batch_len."""
    data = HashableData.default(shape=(4,))
    data = data.replace(id=jnp.array([1, 2, 1, 3]), value=jnp.array([1.0, 2.0, 1.0, 3.0]))
    costs = jnp.array([2.0, 3.0, 1.0, 4.0])

    mask = xnp.unique_mask(data, key=costs, batch_len=4)

    expected_mask = jnp.array([False, True, True, True])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_no_key():
    """Test unique_mask without key returns first occurrence."""
    data = HashableData.default(shape=(5,))
    data = data.replace(id=jnp.array([1, 2, 1, 3, 2]), value=jnp.array([1.0, 2.0, 1.0, 3.0, 2.0]))

    mask = xnp.unique_mask(data, key=None)

    expected_mask = jnp.array([True, True, False, True, False])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_single_element():
    """Test with single element."""
    data = HashableData.default(shape=(1,))
    data = data.replace(id=jnp.array([1]), value=jnp.array([1.0]))

    mask = xnp.unique_mask(data)

    expected_mask = jnp.array([True])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_empty_batch():
    """Test with empty batch."""
    data = HashableData.default(shape=(0,))

    mask = xnp.unique_mask(data, batch_len=0)

    expected_mask = jnp.array([], dtype=jnp.bool_)
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_error_no_uint32ed():
    """Test that error is raised when val doesn't have uint32ed attribute."""
    invalid_data = jnp.array([1, 2, 3])

    with pytest.raises(ValueError, match="key_fn failed to generate hashable keys"):
        xnp.unique_mask(invalid_data)


def test_unique_mask_error_key_length_mismatch():
    """Test that error is raised when key length doesn't match batch_len."""
    data = HashableData.default(shape=(3,))
    data = data.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))
    wrong_key = jnp.array([1.0, 2.0])

    with pytest.raises(ValueError, match="key length 2 must match batch_len 3"):
        xnp.unique_mask(data, key=wrong_key)


def test_unique_mask_complex_scenario():
    """Test a more complex scenario with multiple duplicates and costs."""
    data = HashableData.default(shape=(8,))
    data = data.replace(
        id=jnp.array([1, 2, 3, 1, 2, 3, 1, 4]),
        value=jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 4.0]),
    )
    costs = jnp.array([5.0, 3.0, 4.0, 2.0, 6.0, 1.0, 7.0, 2.0])

    mask = xnp.unique_mask(data, key=costs)

    expected_mask = jnp.array([False, True, False, True, False, True, False, True])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_integration_with_other_ops():
    """Test unique_mask integration with other xnp operations."""
    data = HashableData.default(shape=(6,))
    data = data.replace(
        id=jnp.array([1, 2, 1, 3, 2, 1]),
        value=jnp.array([1.0, 2.0, 1.0, 3.0, 2.0, 1.0]),
    )
    costs = jnp.array([3.0, 2.0, 1.0, 4.0, 5.0, 0.5])

    mask = xnp.unique_mask(data, key=costs)
    filtered_data = xnp.where(mask, data, HashableData.default())

    expected_mask = jnp.array([False, True, False, True, False, True])
    assert jnp.array_equal(mask, expected_mask)

    default_id = HashableData.default().id
    expected_filtered_ids = jnp.where(mask, data.id, default_id)
    assert jnp.array_equal(filtered_data.id, expected_filtered_ids)


def test_unique_mask_with_custom_key_fn():
    """Test unique_mask with custom key function."""
    data = HashableData.default(shape=(6,))
    data = data.replace(
        id=jnp.array([1, 2, 1, 3, 2, 1]),
        value=jnp.array([10.0, 20.0, 10.0, 30.0, 20.0, 10.0]),
    )
    costs = jnp.array([3.0, 2.0, 1.0, 4.0, 5.0, 0.5])

    def custom_key_fn(x):
        return x.value

    mask = xnp.unique_mask(data, key=costs, key_fn=custom_key_fn)

    expected_mask = jnp.array([False, True, False, True, False, True])
    assert jnp.array_equal(mask, expected_mask)
