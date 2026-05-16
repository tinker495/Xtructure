import jax

from xtructure.bgpq._backend import merge_array_backend, select_merge_array_backend
from xtructure.bgpq.merge_split import merge_arrays_parallel, merge_sort_split_idx


def test_merge_backend_policy_uses_sort_split_on_tpu():
    assert select_merge_array_backend("tpu") is merge_sort_split_idx


def test_merge_backend_policy_uses_parallel_for_non_tpu_backends():
    assert select_merge_array_backend("gpu") is merge_arrays_parallel
    assert select_merge_array_backend("cpu") is merge_arrays_parallel


def test_merge_backend_policy_global_matches_current_jax_backend():
    assert merge_array_backend is select_merge_array_backend(jax.default_backend())
