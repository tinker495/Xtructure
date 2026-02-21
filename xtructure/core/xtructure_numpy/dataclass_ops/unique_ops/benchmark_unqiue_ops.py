"""Microbenchmark for unique_mask implementations."""

import argparse
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ....field_descriptors import FieldDescriptor
from ....xtructure_decorators import Xtructurable, xtructure_dataclass
from .legacy_unique_ops import unique_mask_legacy
from .optimized_unique_ops import unique_mask

# -----------------------------------------------------------------------------
# Test Data Structure
# -----------------------------------------------------------------------------


@xtructure_dataclass
class DummyData:
    id: FieldDescriptor.scalar(jnp.uint32)
    category: FieldDescriptor.scalar(jnp.uint8)
    sub_id: FieldDescriptor.scalar(jnp.uint16)


# -----------------------------------------------------------------------------
# Benchmarking Tools
# -----------------------------------------------------------------------------

MethodFn = Callable[..., Any]


def _block_until_ready(outputs: Any) -> None:
    if isinstance(outputs, tuple):
        for out in outputs:
            jax.block_until_ready(out)
    else:
        jax.block_until_ready(outputs)


def _median_iqr_ms(durations: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(durations), dtype=np.float64) * 1000.0
    median = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    return {"median_ms": median, "iqr_ms": float(q75 - q25)}


def _throughput_median(elements: int, durations: Iterable[float]) -> float:
    arr = np.asarray(list(durations), dtype=np.float64)
    # Avoid div by zero
    arr = np.maximum(arr, 1e-9)
    ops = elements / arr
    return float(np.median(ops))


def _make_test_data(
    key_prng: jax.Array,
    size: int,
    duplication_rate: float,
    with_cost: bool,
    with_filled: bool,
    skewed: bool = False,
) -> Tuple[Xtructurable, jax.Array | None, jax.Array | None]:
    """Generate test data with controlled duplication and optional fields."""
    k1, k2, k3, k4, k5 = jax.random.split(key_prng, 5)

    if duplication_rate >= 1.0:
        ids = jnp.zeros(size, dtype=jnp.uint32)
    elif duplication_rate <= 0.0:
        ids = jax.random.permutation(k1, jnp.arange(size, dtype=jnp.uint32))
    else:
        num_unique = int(size * (1 - duplication_rate))
        max_val = 2**30
        unique_pool = jax.random.randint(k1, (num_unique,), 0, max_val, dtype=jnp.uint32)

        if skewed:
            # Skewed distribution: some items are much more frequent
            # Use a power-law like distribution for sampling indices
            p = jnp.exp(-jnp.linspace(0, 5, num_unique))
            p = p / p.sum()
            ids = jax.random.choice(k2, unique_pool, shape=(size,), p=p)
        else:
            # Uniform duplication
            ids = jax.random.choice(k2, unique_pool, shape=(size,))

    categories = jax.random.randint(k3, (size,), 0, 10, dtype=jnp.uint8)
    sub_ids = jax.random.randint(k4, (size,), 0, 1000, dtype=jnp.uint16)

    val = DummyData(id=ids, category=categories, sub_id=sub_ids)

    cost = None
    if with_cost:
        cost = jax.random.uniform(k5, (size,), dtype=jnp.float32)

    filled = None
    if with_filled:
        # 90% filled
        filled = jax.random.bernoulli(k5, 0.9, shape=(size,))

    return val, cost, filled


def _time_method(
    method: MethodFn,
    size: int,
    duplication_rate: float,
    with_cost: bool,
    with_filled: bool,
    skewed: bool,
    trials: int,
    seed: int,
) -> List[float]:
    # JIT compile the method wrapper
    @jax.jit
    def run(v, k, f):
        return method(v, key=k, filled=f)

    # Warmup with one data sample
    w_prng = jax.random.PRNGKey(seed)
    w_val, w_cost, w_filled = _make_test_data(
        w_prng, size, duplication_rate, with_cost, with_filled, skewed
    )
    _block_until_ready(run(w_val, w_cost, w_filled))

    durations: List[float] = []
    for i in range(trials):
        # Per-trial data generation
        trial_prng = jax.random.PRNGKey(seed + i + 1)
        val, cost, filled = _make_test_data(
            trial_prng, size, duplication_rate, with_cost, with_filled, skewed
        )

        start = time.perf_counter()
        out = run(val, cost, filled)
        _block_until_ready(out)
        durations.append(time.perf_counter() - start)
    return durations


def run_bench(
    sizes: List[int],
    duplication_rates: List[float],
    with_cost: bool,
    with_filled: bool,
    skewed: bool,
    trials: int,
    warmup: int,
    seed: int,
) -> None:
    print("Unique Mask Microbench (Refined)")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Trials: {trials}, Skewed: {skewed}")
    print(f"Settings: Cost={with_cost}, Filled={with_filled}")
    print("-" * 80)

    methods = {
        "legacy": unique_mask_legacy,
        "lexsort": unique_mask,
    }

    winners: Dict[Tuple[int, float], str] = {}

    for size in sizes:
        for dup in duplication_rates:
            print(f"\nSize={size}, Duplication={dup * 100:.0f}%")

            results = {}
            throughputs = {}

            for name, method in methods.items():
                durations = _time_method(
                    method, size, dup, with_cost, with_filled, skewed, trials, seed
                )
                stats = _median_iqr_ms(durations)
                results[name] = stats
                throughputs[name] = _throughput_median(size, durations)

                print(
                    f"  {name:10s} median={stats['median_ms']:8.4f} ms  "
                    f"iqr={stats['iqr_ms']:7.4f} ms  "
                    f"ops/s={throughputs[name]:10.2e}"
                )

            winner = min(results.items(), key=lambda x: x[1]["median_ms"])[0]
            winners[(size, dup)] = winner

            legacy_ms = results["legacy"]["median_ms"]
            lexsort_ms = results["lexsort"]["median_ms"]
            if legacy_ms > 0 and lexsort_ms > 0:
                speedup = legacy_ms / lexsort_ms
                percentage = (speedup - 1) * 100
                if speedup >= 1.0:
                    print(
                        f"  Speedup (Legacy -> Lexsort): {speedup:.2f}x ({percentage:.1f}% faster)"
                    )
                else:
                    slowdown = (1 / speedup - 1) * 100
                    print(f"  Speedup (Legacy -> Lexsort): {speedup:.2f}x ({slowdown:.1f}% slower)")

    print("\nSummary (Winner by Setting):")
    for (size, dup), winner in winners.items():
        print(f"  Size={size:<8} Dup={dup * 100:<3.0f}% : {winner}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark unique_mask implementations.")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[4096, 16384, 65536], help="Batch sizes to test."
    )
    parser.add_argument(
        "--duplication",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 0.9],
        help="Duplication rates (0.0 to 1.0).",
    )
    parser.add_argument("--no-cost", action="store_true", help="Disable cost/key array.")
    parser.add_argument("--no-filled", action="store_true", help="Disable filled mask.")
    parser.add_argument("--skewed", action="store_true", help="Enable skewed distribution.")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_bench(
        sizes=args.sizes,
        duplication_rates=args.duplication,
        with_cost=not args.no_cost,
        with_filled=not args.no_filled,
        skewed=args.skewed,
        trials=args.trials,
        warmup=args.warmup,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
