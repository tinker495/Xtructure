"""GPU singleton-group deduplication and representative selection."""

import jax
import jax.numpy as jnp
import pytest

from xtructure.hashtable.hash_utils import (
    _compute_unique_mask_from_uint32eds,
    _unique_groups_exact,
    _unique_mask_from_group_ids,
)


def _reference(uint32eds, filled, unique_key):
    """The pre-fastpath implementation: exact grouping + representative pick."""
    inverse = _unique_groups_exact(uint32eds, filled)
    return _unique_mask_from_group_ids(inverse, filled, unique_key)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="GPU-only shape gate")
def test_gpu_large_singleton_groups_skip_group_materialization():
    batch_len = 4096
    rows = jnp.arange(batch_len * 8, dtype=jnp.uint32).reshape(batch_len, 8)
    filled = jnp.ones((batch_len,), dtype=jnp.bool_)
    row_hash = jnp.arange(batch_len, dtype=jnp.uint32)

    got_mask, got_rep = _compute_unique_mask_from_uint32eds(rows, filled, None, row_hash=row_hash)

    assert bool(jnp.all(got_mask))
    assert bool(jnp.all(got_rep == jnp.arange(batch_len, dtype=jnp.int32)))

    unique_key = jnp.zeros((batch_len,), dtype=jnp.float32).at[1].set(jnp.nan)
    keyed_mask, keyed_rep = _compute_unique_mask_from_uint32eds(
        rows, filled, unique_key, row_hash=row_hash
    )
    expected_mask, expected_rep = _reference(rows, filled, unique_key)
    assert bool(jnp.all(keyed_mask == expected_mask))
    assert bool(jnp.all(keyed_rep == expected_rep))
