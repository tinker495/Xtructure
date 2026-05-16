import jax
import jax.numpy as jnp

import xtructure.bgpq._backend as backend_policy
from xtructure.bgpq._backend import (
    PARALLEL_MERGE_MIN_ELEMENTS,
    merge_array_backend,
    merge_arrays_adaptive,
    select_merge_array_backend,
)
from xtructure.bgpq.merge_split import merge_arrays_parallel, merge_sort_split_idx


def test_merge_backend_policy_uses_sort_split_on_tpu():
    assert select_merge_array_backend("tpu") is merge_sort_split_idx


def test_merge_backend_policy_uses_parallel_for_non_tpu_backends():
    assert select_merge_array_backend("gpu") is merge_arrays_adaptive
    assert select_merge_array_backend("cpu") is merge_arrays_adaptive


def test_merge_backend_policy_global_matches_current_jax_backend():
    assert merge_array_backend is select_merge_array_backend(jax.default_backend())


def test_adaptive_backend_preserves_merge_order_below_parallel_threshold():
    keys_a = jnp.array([1.0, 3.0], dtype=jnp.float32)
    keys_b = jnp.array([2.0, 4.0], dtype=jnp.float32)

    merged, indices = merge_arrays_adaptive(keys_a, keys_b)

    assert keys_a.shape[0] + keys_b.shape[0] < PARALLEL_MERGE_MIN_ELEMENTS
    assert jnp.all(merged == jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32))
    assert jnp.all(indices == jnp.array([0, 2, 1, 3], dtype=jnp.int32))


def test_parallel_backend_remains_available_for_large_merges():
    assert merge_arrays_parallel is not merge_sort_split_idx


def test_adaptive_backend_cutover_uses_parallel_at_threshold(monkeypatch):
    calls = []

    def fake_split(ak, bk):
        calls.append("split")
        return ak, jnp.arange(ak.shape[0] + bk.shape[0], dtype=jnp.int32)

    def fake_parallel(ak, bk):
        calls.append("parallel")
        return ak, jnp.arange(ak.shape[0] + bk.shape[0], dtype=jnp.int32)

    monkeypatch.setattr(backend_policy, "merge_sort_split_idx", fake_split)
    monkeypatch.setattr(backend_policy, "merge_arrays_parallel", fake_parallel)

    below = PARALLEL_MERGE_MIN_ELEMENTS // 2 - 1
    at_cutover = PARALLEL_MERGE_MIN_ELEMENTS // 2

    backend_policy.merge_arrays_adaptive(jnp.ones((below,)), jnp.ones((below,)))
    backend_policy.merge_arrays_adaptive(jnp.ones((at_cutover,)), jnp.ones((at_cutover,)))

    assert calls == ["split", "parallel"]
