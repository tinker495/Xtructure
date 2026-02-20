import jax.numpy as jnp

from tests.testdata import HeapValueABC
from xtructure import BGPQ
from xtructure.bgpq._delete import _bgpq_delete_mins_jit


def test_delete_mins_direct_coverage():
    batch_size = 4
    total_size = 4
    heap = BGPQ.build(total_size, batch_size, HeapValueABC, jnp.float32)
    k1 = jnp.full((batch_size,), 100.0, dtype=jnp.float32)
    v1 = HeapValueABC.default((batch_size,))
    v1.a = jnp.full((batch_size,), 1, dtype=jnp.uint8)

    heap = heap.insert(k1, v1)

    # call internal delete to satisfy direct import and coverage metric
    heap, min_key, min_val = _bgpq_delete_mins_jit(heap)

    assert jnp.all(min_key == 100.0)
    assert jnp.all(min_val.a == 1)
