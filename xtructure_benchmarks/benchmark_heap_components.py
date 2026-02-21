import argparse
import os
from typing import Tuple

import jax
import jax.numpy as jnp

from tests.testdata import HeapValueABC
from xtructure import BGPQ
from xtructure.bgpq._merge import merge_sort_split
from xtructure.bgpq._utils import sort_arrays
from xtructure_benchmarks.common import (
    check_system_load,
    run_jax_trials,
    throughput_stats,
)


@jax.jit
def key_gen(x: HeapValueABC) -> jax.Array:
    uint32_hash = x.hash()
    key = uint32_hash % (2**12) / (2**8)
    return jnp.asarray(key, dtype=jnp.float32)


vmapped_key_gen = jax.jit(jax.vmap(key_gen))


def _make_sorted_block(
    key: jax.Array, batch_size: int
) -> Tuple[jax.Array, HeapValueABC]:
    values = HeapValueABC.random(shape=(batch_size,), key=key)
    keys = vmapped_key_gen(values)
    return sort_arrays(keys, values)


def _bench(name: str, fn, args_supplier, num_ops: int, trials: int, warmup: int):
    durations, mem = run_jax_trials(
        fn, trials=trials, warmup=warmup, args_supplier=args_supplier
    )
    stats = throughput_stats(num_ops, (durations, mem))
    print(f"{name:24s} median={stats['median']:.2f} iqr={stats['iqr']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BGPQ component costs.")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    check_system_load()
    batch_size = args.batch_size
    key_a, key_b = jax.random.split(jax.random.PRNGKey(batch_size), 2)
    ak, av = _make_sorted_block(key_a, batch_size)
    bk, bv = _make_sorted_block(key_b, batch_size)

    print("BGPQ component benchmarks")
    print(f"batch_size={batch_size} trials={args.trials} warmup={args.warmup}")
    print(f"value_reorder={os.environ.get('XTRUCTURE_BGPQ_VALUE_REORDER', 'gather')}")
    print(f"kv_backend={os.environ.get('XTRUCTURE_BGPQ_MERGE_VALUE_BACKEND', 'off')}")

    _bench(
        "sort_arrays",
        sort_arrays,
        lambda: (ak, av),
        batch_size,
        args.trials,
        args.warmup,
    )
    _bench(
        "merge_sort_split",
        merge_sort_split,
        lambda: (ak, av, bk, bv),
        batch_size * 2,
        args.trials,
        args.warmup,
    )

    heap = BGPQ.build(batch_size * 2, batch_size, HeapValueABC, jnp.float32)
    _bench(
        "heap.insert",
        heap.insert,
        lambda: (ak, av),
        batch_size,
        args.trials,
        args.warmup,
    )


if __name__ == "__main__":
    main()
