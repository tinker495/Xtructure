import argparse
import time
from typing import Callable, Dict, Iterable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .loop import merge_arrays_indices_loop
from .parallel import merge_arrays_parallel
from .split import merge_sort_split_idx

MethodFn = Callable[[jax.Array, jax.Array], Tuple[jax.Array, jax.Array]]

# Empirical results (GPU, float32, trials=10000, warmup=3):
# n=m=1024:  loop 1.383 ms, parallel 0.219 ms, split 1.421 ms
# n=m=4096:  loop 1.383 ms, parallel 0.195 ms, split 1.391 ms
# n=m=16384: loop 2.505 ms, parallel 0.181 ms, split 1.383 ms
# Winner across sizes: parallel


def _block_until_ready(outputs: Tuple[jax.Array, jax.Array]) -> None:
    keys, indices = outputs
    jax.block_until_ready(keys)
    jax.block_until_ready(indices)


def _median_iqr_ms(durations: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(durations), dtype=np.float64) * 1000.0
    median = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    return {"median_ms": median, "iqr_ms": float(q75 - q25)}


def _throughput_median(elements: int, durations: Iterable[float]) -> float:
    arr = np.asarray(list(durations), dtype=np.float64)
    ops = elements / arr
    return float(np.median(ops))


def _make_sorted_arrays(key: jax.Array, size: int, dtype: jnp.dtype) -> Tuple[jax.Array, jax.Array]:
    key_a, key_b = jax.random.split(key, 2)
    a = jax.random.uniform(key_a, shape=(size,), dtype=dtype)
    b = jax.random.uniform(key_b, shape=(size,), dtype=dtype)
    return jnp.sort(a), jnp.sort(b)


def _time_method(method: MethodFn, ak: jax.Array, bk: jax.Array, trials: int) -> List[float]:
    durations: List[float] = []
    for _ in range(trials):
        start = time.perf_counter()
        outputs = method(ak, bk)
        _block_until_ready(outputs)
        durations.append(time.perf_counter() - start)
    return durations


def _print_size_header(size: int) -> None:
    total = size * 2
    print(f"\nSize n=m={size} (total={total})")


def _print_method_line(name: str, stats: Dict[str, float], throughput: float) -> None:
    median_ms = stats["median_ms"]
    iqr_ms = stats["iqr_ms"]
    print(
        f"  {name:12s} median={median_ms:8.3f} ms  iqr={iqr_ms:7.3f} ms  ops/s={throughput:10.2f}"
    )


def run_bench(
    sizes: List[int],
    trials: int,
    warmup: int,
    dtype: jnp.dtype,
    methods: Dict[str, MethodFn],
    seed: int,
) -> None:
    dtype_name = jnp.dtype(dtype).name
    print("Merge/split microbench")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Trials: {trials}, warmup: {warmup}, dtype: {dtype_name}")
    print("Methods:", ", ".join(methods.keys()))

    winners: Dict[int, str] = {}
    for size in sizes:
        _print_size_header(size)
        key = jax.random.PRNGKey(seed + size)
        ak, bk = _make_sorted_arrays(key, size, dtype)
        total = ak.shape[0] + bk.shape[0]

        for _ in range(warmup):
            for method in methods.values():
                _block_until_ready(method(ak, bk))

        results: Dict[str, Dict[str, float]] = {}
        throughput: Dict[str, float] = {}
        for name, method in methods.items():
            durations = _time_method(method, ak, bk, trials)
            stats = _median_iqr_ms(durations)
            results[name] = stats
            throughput[name] = _throughput_median(total, durations)
            _print_method_line(name, stats, throughput[name])

        winner = min(results.items(), key=lambda item: item[1]["median_ms"])[0]
        winners[size] = winner
        print(f"  winner: {winner}")

    print("\nSummary (winner by size):")
    for size in sizes:
        print(f"  n=m={size}: {winners[size]}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat merge/split experiments to compare method speed."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[2**10, 2**12, 2**14],
        help="List of per-array sizes (n=m).",
    )
    parser.add_argument("--trials", type=int, default=10000, help="Number of timed trials.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Input dtype for the merge inputs.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["loop", "parallel", "split"],
        choices=["loop", "parallel", "split"],
        help="Methods to benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dtype_map = {
        "float16": jnp.float16,
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    method_map: Dict[str, MethodFn] = {
        "loop": merge_arrays_indices_loop,
        "parallel": merge_arrays_parallel,
        "split": merge_sort_split_idx,
    }
    methods = {name: method_map[name] for name in args.methods}

    run_bench(
        sizes=args.sizes,
        trials=args.trials,
        warmup=args.warmup,
        dtype=dtype,
        methods=methods,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
