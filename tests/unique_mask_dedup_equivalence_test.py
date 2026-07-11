"""Dedup fast path must be bit-equivalent to exact full-row unique grouping.

`_compute_unique_mask_from_uint32eds` groups by a u32 row hash and falls back
to the exact path on any verified hash collision; these tests pin the output
contract (unique_mask, representative_indices) against a reference
implementation of the exact path across randomized inputs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtructure.hashtable.hash_utils import (
    _compute_unique_mask_from_uint32eds,
    _unique_mask_from_group_ids,
    _unique_groups_exact,
)


def _reference(uint32eds, filled, unique_key):
    """The pre-fastpath implementation: exact grouping + representative pick."""
    inverse = _unique_groups_exact(uint32eds, filled)
    return _unique_mask_from_group_ids(inverse, filled, unique_key)


def _random_case(rng, n, lanes, dup_frac, fill_frac, with_key):
    base = rng.integers(0, 2**32, size=(n, lanes), dtype=np.uint32)
    n_dup = int(n * dup_frac)
    if n_dup:
        src = rng.integers(0, n, size=n_dup)
        dst = rng.integers(0, n, size=n_dup)
        base[dst] = base[src]
    filled = rng.random(n) < fill_frac
    key = rng.random(n).astype(np.float32) if with_key else None
    return (
        jnp.asarray(base),
        jnp.asarray(filled),
        None if key is None else jnp.asarray(key),
    )


@pytest.mark.parametrize("with_key", [False, True])
def test_fastpath_matches_exact_reference_randomized(with_key):
    rng = np.random.default_rng(7 if with_key else 3)
    for trial in range(60):
        n = int(rng.integers(2, 257))
        lanes = int(rng.integers(1, 12))
        dup_frac = float(rng.choice([0.0, 0.3, 0.9]))
        fill_frac = float(rng.choice([0.0, 0.5, 1.0]))
        rows, filled, key = _random_case(rng, n, lanes, dup_frac, fill_frac, with_key)

        got_mask, got_rep = _compute_unique_mask_from_uint32eds(rows, filled, key)
        want_mask, want_rep = _reference(rows, filled, key)

        assert bool(jnp.all(got_mask == want_mask)), f"mask diverged (trial {trial})"
        assert bool(jnp.all(got_rep == want_rep)), f"representatives diverged (trial {trial})"


def test_fastpath_handles_all_duplicates_and_sentinel_content():
    # every row identical AND equal to the unfilled sentinel pattern
    rows = jnp.full((16, 4), jnp.uint32(0xFFFFFFFF))
    filled = jnp.asarray([True] * 8 + [False] * 8)
    got_mask, got_rep = _compute_unique_mask_from_uint32eds(rows, filled, None)
    want_mask, want_rep = _reference(rows, filled, None)
    assert bool(jnp.all(got_mask == want_mask))
    assert bool(jnp.all(got_rep == want_rep))
    assert int(got_mask.sum()) == 1  # one representative among filled dups


def test_collision_fallback_is_exact():
    """Force the hash-collision path by monkeypatching the row hash to a
    constant: distinct rows then share one hash group, verification must
    fail, and the lax.cond fallback must reproduce exact grouping."""
    import xtructure.hashtable.hash_utils as hu

    rows = jnp.asarray(np.random.default_rng(0).integers(0, 2**32, (64, 6), dtype=np.uint32))
    filled = jnp.ones(64, dtype=jnp.bool_)

    original = hu.hash_fast_uint32ed_batched
    hu.hash_fast_uint32ed_batched = lambda r, s: jnp.zeros(r.shape[0], dtype=jnp.uint32)
    try:
        got_mask, got_rep = _compute_unique_mask_from_uint32eds(rows, filled, None)
    finally:
        hu.hash_fast_uint32ed_batched = original

    want_mask, want_rep = _reference(rows, filled, None)
    assert bool(jnp.all(got_mask == want_mask))
    assert bool(jnp.all(got_rep == want_rep))


def test_one_dimensional_rows_supported():
    rows = jnp.asarray([5, 5, 3, 7, 3, 5], dtype=jnp.uint32)
    filled = jnp.ones(6, dtype=jnp.bool_)
    got_mask, got_rep = _compute_unique_mask_from_uint32eds(rows, filled, None)
    assert got_mask.tolist() == [True, False, True, True, False, False]
    assert got_rep.tolist() == [0, 0, 2, 3, 2, 0]
