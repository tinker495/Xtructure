import argparse
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import jax
import jax.numpy as jnp

from xtructure import BGPQ, FieldDescriptor, Xtructurable, xtructure_dataclass
from xtructure_benchmarks.common import (
    check_system_load,
    get_system_info,
    run_jax_trials,
    throughput_stats,
)

if TYPE_CHECKING:

    class BenchmarkValueSmall(Xtructurable):
        a: FieldDescriptor
        b: FieldDescriptor
        c: FieldDescriptor

else:

    @xtructure_dataclass
    class BenchmarkValueSmall:
        a: FieldDescriptor(jnp.uint8)  # type: ignore
        b: FieldDescriptor(jnp.uint32, (1, 2))  # type: ignore
        c: FieldDescriptor(jnp.float32, (1, 2, 3))  # type: ignore


@jax.jit
def key_gen(x: BenchmarkValueSmall) -> jax.Array:
    uint32_hash = x.hash()
    key = uint32_hash % (2**12) / (2**8)
    return jnp.asarray(key, dtype=jnp.float32)


vmapped_key_gen = jax.jit(jax.vmap(key_gen))


@jax.jit
def _insert_loop(heap: BGPQ, keys: jnp.ndarray, values: BenchmarkValueSmall):
    def body(idx, h):
        key_i = keys[idx]
        val_i = jax.tree_util.tree_map(lambda leaf: leaf[idx], values)
        return h.insert(key_i, val_i)

    return jax.lax.fori_loop(0, keys.shape[0], body, heap)


@jax.jit
def _delete_loop(heap: BGPQ, count: jnp.ndarray):
    def body(_, h):
        h, _, _ = BGPQ.delete_mins(h)
        return h

    return jax.lax.fori_loop(0, count, body, heap)


def _bench_insert(
    insert_fn,
    trials: int,
    args_supplier,
) -> Tuple[List[float], List[float]]:
    durations, peak_mem = run_jax_trials(
        insert_fn,
        trials=trials,
        args_supplier=args_supplier,
    )
    return durations, peak_mem


def _profile_run(profile_dir: str, tag: str, func, args):
    trace_dir = os.path.join(profile_dir, tag)
    os.makedirs(trace_dir, exist_ok=True)
    jax.profiler.start_trace(trace_dir)
    result = func(*args)
    jax.block_until_ready(result)
    jax.profiler.stop_trace()


def _bench_delete(
    delete_fn,
    trials: int,
    include_host_transfer: bool,
    args_supplier,
) -> Tuple[List[float], List[float]]:
    durations, peak_mem = run_jax_trials(
        delete_fn,
        trials=trials,
        include_device_transfer=include_host_transfer,
        args_supplier=args_supplier,
    )
    return durations, peak_mem


def run_benchmarks(
    trials: int,
    batch_sizes: List[int],
    mode: str,
    num_inserts: int,
    prefill: int,
    bench: str,
    profile_dir: Optional[str],
) -> Dict[str, Any]:
    check_system_load()

    results: Dict[str, Any] = {
        "batch_sizes": batch_sizes,
        "xtructure": {},
        "environment": get_system_info(),
        "config": {
            "merge_value_backend": os.environ.get("XTRUCTURE_BGPQ_MERGE_VALUE_BACKEND", ""),
            "merge_value_backend_buffer": os.environ.get(
                "XTRUCTURE_BGPQ_MERGE_VALUE_BACKEND_BUFFER", ""
            ),
            "merge_value_backend_sortsplit": os.environ.get(
                "XTRUCTURE_BGPQ_MERGE_VALUE_BACKEND_SORTSPLIT", ""
            ),
            "merge_value_packing": os.environ.get("XTRUCTURE_BGPQ_MERGE_VALUE_PACKING", "auto"),
            "merge_value_scalar_max": os.environ.get("XTRUCTURE_BGPQ_MERGE_VALUE_SCALAR_MAX", "16"),
            "merge_value_auto_min_batch": os.environ.get(
                "XTRUCTURE_BGPQ_MERGE_VALUE_AUTO_MIN_BATCH", "0"
            ),
            "merge_value_reorder": os.environ.get("XTRUCTURE_BGPQ_VALUE_REORDER", "gather"),
            "num_inserts": num_inserts,
            "prefill": prefill,
            "bench": bench,
            "profile_dir": profile_dir or "",
        },
    }

    print("Running Heap (BGPQ) Small-Value Benchmarks...")
    print(f"JAX backend: {jax.default_backend()}")

    for batch_size in batch_sizes:
        print(f"  Batch Size: {batch_size}")
        bench_cls = cast(Any, BenchmarkValueSmall)
        rng = jax.random.PRNGKey(batch_size)
        rng, insert_key, prefill_key, delete_key = jax.random.split(rng, 4)

        insert_values = bench_cls.random(shape=(num_inserts, batch_size), key=insert_key)
        insert_keys = jax.vmap(vmapped_key_gen)(insert_values)

        prefill_values = None
        prefill_keys = None
        if prefill > 0:
            prefill_values = bench_cls.random(shape=(prefill, batch_size), key=prefill_key)
            prefill_keys = jax.vmap(vmapped_key_gen)(prefill_values)

        delete_prefill = max(prefill, num_inserts) if bench in {"delete", "both"} else prefill
        delete_values = None
        delete_keys = None
        if bench in {"delete", "both"}:
            delete_values = bench_cls.random(shape=(delete_prefill, batch_size), key=delete_key)
            delete_keys = jax.vmap(vmapped_key_gen)(delete_values)

        max_batches = max(num_inserts, prefill, delete_prefill)
        max_size = int(batch_size * (max_batches + 1))
        heap = BGPQ.build(max_size, batch_size, bench_cls, jnp.float32)

        prefill_heap = heap
        if prefill > 0:
            prefill_heap = _insert_loop(heap, prefill_keys, prefill_values)
            jax.block_until_ready(prefill_heap)

        if bench in {"insert", "both"}:
            if mode == "e2e":
                insert_keys_host = jax.device_get(insert_keys)
                insert_values_host = jax.device_get(insert_values)

                def insert_args_supplier():
                    return (
                        jax.device_put(insert_keys_host),
                        jax.device_put(insert_values_host),
                    )

            else:

                def insert_args_supplier():
                    return (insert_keys, insert_values)

            def insert_fn(keys, values):
                return _insert_loop(prefill_heap, keys, values)

            if profile_dir:
                insert_fn(*insert_args_supplier())
                _profile_run(
                    profile_dir,
                    f"insert_batch{batch_size}",
                    insert_fn,
                    insert_args_supplier(),
                )
            insert_durations, insert_mem = _bench_insert(
                insert_fn,
                trials=trials,
                args_supplier=insert_args_supplier,
            )
            insert_stats = throughput_stats(
                batch_size * num_inserts, (insert_durations, insert_mem)
            )
            results["xtructure"].setdefault("insert_ops_per_sec", []).append(insert_stats)

        if bench in {"delete", "both"}:
            heap_delete = heap
            if delete_prefill > 0:
                heap_delete = _insert_loop(heap, delete_keys, delete_values)
                jax.block_until_ready(heap_delete)

            delete_count = jnp.asarray(num_inserts, dtype=jnp.int32)

            def delete_fn():
                return _delete_loop(heap_delete, delete_count)

            if profile_dir:
                delete_fn()
                _profile_run(
                    profile_dir,
                    f"delete_batch{batch_size}",
                    lambda h: _delete_loop(h, delete_count),
                    (heap_delete,),
                )
            delete_durations, delete_mem = _bench_delete(
                delete_fn,
                trials=trials,
                include_host_transfer=(mode == "e2e"),
                args_supplier=None,
            )
            delete_stats = throughput_stats(
                batch_size * num_inserts, (delete_durations, delete_mem)
            )
            results["xtructure"].setdefault("delete_ops_per_sec", []).append(delete_stats)

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BGPQ with small value payloads.")
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of timed trials.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["kernel", "e2e"],
        default="kernel",
        help="kernel: device-only, e2e: include host transfer",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[2**10, 2**12, 2**14],
        help="Batch sizes to benchmark.",
    )
    parser.add_argument(
        "--num-inserts",
        type=int,
        default=1,
        help="Number of insert batches per trial.",
    )
    parser.add_argument(
        "--prefill",
        type=int,
        default=0,
        help="Number of batches to prefill before timing.",
    )
    parser.add_argument(
        "--bench",
        type=str,
        choices=["insert", "delete", "both"],
        default="both",
        help="Which operations to benchmark.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="",
        help="Directory to write JAX profiler traces (empty to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile_dir = args.profile_dir or None
    results = run_benchmarks(
        args.trials,
        args.batch_sizes,
        args.mode,
        args.num_inserts,
        args.prefill,
        args.bench,
        profile_dir,
    )
    print("Results:")
    for idx, batch_size in enumerate(results["batch_sizes"]):
        parts = [f"  batch={batch_size}"]
        if "insert_ops_per_sec" in results["xtructure"]:
            insert_stats = results["xtructure"]["insert_ops_per_sec"][idx]
            parts.append(
                f"insert median={insert_stats['median']:.2f} iqr={insert_stats['iqr']:.2f}"
            )
        if "delete_ops_per_sec" in results["xtructure"]:
            delete_stats = results["xtructure"]["delete_ops_per_sec"][idx]
            parts.append(
                f"delete median={delete_stats['median']:.2f} iqr={delete_stats['iqr']:.2f}"
            )
        print(" | ".join(parts))


if __name__ == "__main__":
    main()
