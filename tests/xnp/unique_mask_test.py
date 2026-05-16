"""Tests for xnp.unique_mask helper."""

import jax
import jax.numpy as jnp
import pytest

from tests.xnp.shared_data import HashableData
from xtructure import FieldDescriptor
from xtructure import numpy as xnp
from xtructure import xtructure_dataclass
from xtructure.core.xtructure_numpy.dataclass_ops.unique_ops.optimized_unique_ops import (
    _batched_uint32_keys,
)


@xtructure_dataclass
class MixedWidthHashData:
    """Dataclass covering sub-32-bit leaves in default unique key generation."""

    flag: FieldDescriptor.scalar(dtype=jnp.bool_)
    code: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(3,))
    pair: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(3,))
    score: FieldDescriptor.scalar(dtype=jnp.float16)


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


def test_unique_mask_custom_float_key_preserves_fractional_values():
    """Custom float keys must not collide through uint32 truncation."""
    data = HashableData.default(shape=(4,))
    data = data.replace(
        id=jnp.array([1, 2, 3, 4]),
        value=jnp.array([1.25, 1.75, 1.25, 2.5]),
    )
    costs = jnp.array([3.0, 1.0, 2.0, 4.0])

    def custom_key_fn(x):
        return x.value

    mask = xnp.unique_mask(data, key=costs, key_fn=custom_key_fn)

    expected_mask = jnp.array([False, True, True, True])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_return_index_is_jit_static():
    """return_index should keep a static batch-sized shape under JIT."""
    data = HashableData.default(shape=(5,))
    data = data.replace(
        id=jnp.array([1, 2, 1, 3, 2]),
        value=jnp.array([1.0, 2.0, 1.0, 3.0, 2.0]),
    )

    @jax.jit
    def run(v):
        return xnp.unique_mask(v, return_index=True)

    mask, indices = run(data)

    expected_mask = jnp.array([True, True, False, True, False])
    assert jnp.array_equal(mask, expected_mask)
    assert indices.shape == (5,)
    assert jnp.array_equal(
        jnp.sort(indices[indices < 5]),
        jnp.array([0, 1, 3], dtype=indices.dtype),
    )


def test_unique_mask_return_index_with_cost_is_jit_static():
    """Cost-based return_index should select winning rows without dynamic indexing."""
    data = HashableData.default(shape=(6,))
    data = data.replace(
        id=jnp.array([1, 2, 1, 3, 2, 1]),
        value=jnp.array([1.0, 2.0, 1.0, 3.0, 2.0, 1.0]),
    )
    costs = jnp.array([5.0, 3.0, 2.0, 4.0, 7.0, 1.0])

    @jax.jit
    def run(v, k):
        return xnp.unique_mask(v, key=k, return_index=True)

    mask, indices = run(data, costs)

    expected_mask = jnp.array([False, True, False, True, False, True])
    assert jnp.array_equal(mask, expected_mask)
    assert indices.shape == (6,)
    assert jnp.array_equal(
        jnp.sort(indices[indices < 6]),
        jnp.array([1, 3, 5], dtype=indices.dtype),
    )


def test_unique_mask_default_key_matches_scalar_uint32ed_for_mixed_width_leaves():
    """Batched default key generation should match per-row uint32ed semantics."""
    data = MixedWidthHashData.default(shape=(4,))
    data = data.replace(
        flag=jnp.array([True, False, True, False]),
        code=jnp.array(
            [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]],
            dtype=jnp.uint8,
        ),
        pair=jnp.array(
            [[7, 8, 9], [10, 11, 12], [7, 8, 9], [10, 11, 12]],
            dtype=jnp.uint16,
        ),
        score=jnp.array([1.5, 2.5, 1.5, 2.5], dtype=jnp.float16),
    )

    batched_keys = _batched_uint32_keys(data, batch_len=4)
    scalar_keys = jax.vmap(lambda x: x.uint32ed)(data)

    assert jnp.array_equal(batched_keys, scalar_keys)
    assert jnp.array_equal(xnp.unique_mask(data), jnp.array([True, True, False, False]))
