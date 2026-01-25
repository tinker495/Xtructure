import argparse
import itertools
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np

from xtructure import FieldDescriptor, xtructure_dataclass

from ...core import Xtructurable
from ...core import xtructure_numpy as xnp
from .loop import merge_arrays_indices_loop
from .parallel import (
    _merge_parallel_config,
    merge_arrays_parallel,
    merge_arrays_parallel_kv,
)
from .split import merge_sort_split_idx

MethodFn = Callable[[chex.Array, chex.Array], Tuple[chex.Array, chex.Array]]
ValueMethodFn = Callable[[chex.Array, Xtructurable, chex.Array, Xtructurable], Tuple]

# Note: microbench results are hardware/driver/version dependent.


@xtructure_dataclass
class BenchValue:
    a: FieldDescriptor(jnp.uint8)  # type: ignore
    b: FieldDescriptor(jnp.uint32, (1, 2))  # type: ignore
    c: FieldDescriptor(jnp.float32, (1, 2, 3))  # type: ignore


def _block_until_ready(outputs) -> None:
    for leaf in jax.tree_util.tree_leaves(outputs):
        jax.block_until_ready(leaf)


def _assert_device_true(condition: jax.Array, message: str) -> None:
    if not bool(jax.device_get(condition)):
        raise AssertionError(message)


def _assert_tree_equal(lhs, rhs, message: str) -> None:
    lhs_leaves = jax.tree_util.tree_leaves(lhs)
    rhs_leaves = jax.tree_util.tree_leaves(rhs)
    if len(lhs_leaves) != len(rhs_leaves):
        raise AssertionError(message)
    for left, right in zip(lhs_leaves, rhs_leaves):
        _assert_device_true(jnp.array_equal(left, right), message)


def _verify_outputs(
    name: str,
    outputs: Tuple[chex.Array, chex.Array],
    ak: chex.Array,
    bk: chex.Array,
    reference: Tuple[chex.Array, chex.Array],
) -> None:
    keys, indices = outputs
    concat = jnp.concatenate([ak, bk])
    total_len = concat.shape[0]

    in_bounds = jnp.logical_and(indices >= 0, indices < total_len)
    _assert_device_true(jnp.all(in_bounds), f"{name}: indices out of bounds")

    reconstructed = concat[indices]
    _assert_device_true(jnp.array_equal(keys, reconstructed), f"{name}: keys do not match indices")

    keys_any = cast(Any, keys)
    sorted_ok = jnp.all(keys_any[:-1] <= keys_any[1:])
    _assert_device_true(sorted_ok, f"{name}: keys are not sorted")

    ref_keys, ref_indices = reference
    _assert_device_true(jnp.array_equal(keys, ref_keys), f"{name}: keys differ from reference")
    _assert_device_true(
        jnp.array_equal(indices, ref_indices), f"{name}: indices differ from reference"
    )


def _verify_value_outputs(name: str, outputs: Tuple, reference: Tuple) -> None:
    _assert_tree_equal(outputs, reference, f"{name}: outputs differ from reference")


def _median_iqr_ms(durations: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(durations), dtype=np.float64) * 1000.0
    median = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    return {"median_ms": median, "iqr_ms": float(q75 - q25)}


def _throughput_median(elements: int, durations: Iterable[float]) -> float:
    arr = np.asarray(list(durations), dtype=np.float64)
    ops = elements / arr
    return float(np.median(ops))


def _make_sorted_arrays(
    key: jax.Array, size_a: int, size_b: int, dtype: jnp.dtype
) -> Tuple[chex.Array, chex.Array]:
    key_a, key_b = jax.random.split(key, 2)
    a = jax.random.uniform(key_a, shape=(size_a,), dtype=dtype)
    b = jax.random.uniform(key_b, shape=(size_b,), dtype=dtype)
    return jnp.sort(a), jnp.sort(b)


def _make_sorted_inputs(
    key: jax.Array, size_a: int, size_b: int, dtype: jnp.dtype, value_cls
) -> Tuple[jax.Array, jax.Array, Xtructurable, Xtructurable]:
    key_a, key_b, val_a_key, val_b_key = jax.random.split(key, 4)
    ak = jax.random.uniform(key_a, shape=(size_a,), dtype=dtype)
    bk = jax.random.uniform(key_b, shape=(size_b,), dtype=dtype)
    av = value_cls.random(shape=(size_a,), key=val_a_key)
    bv = value_cls.random(shape=(size_b,), key=val_b_key)
    return jnp.sort(ak), jnp.sort(bk), av, bv


def _time_method(method: MethodFn, ak: chex.Array, bk: chex.Array, trials: int) -> List[float]:
    durations: List[float] = []
    for _ in range(trials):
        start = time.perf_counter()
        outputs = method(ak, bk)
        _block_until_ready(outputs)
        durations.append(time.perf_counter() - start)
    return durations


def _time_method_values(
    method: ValueMethodFn,
    ak: jax.Array,
    av: Xtructurable,
    bk: jax.Array,
    bv: Xtructurable,
    trials: int,
) -> List[float]:
    durations: List[float] = []
    for _ in range(trials):
        start = time.perf_counter()
        outputs = method(ak, av, bk, bv)
        _block_until_ready(outputs)
        durations.append(time.perf_counter() - start)
    return durations


def _print_size_header(size_a: int, size_b: int) -> None:
    total = size_a + size_b
    if size_a == size_b:
        print(f"\nSize n=m={size_a} (total={total})")
    else:
        print(f"\nSize n={size_a} m={size_b} (total={total})")


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
    verify: bool,
    size_offset: int,
) -> None:
    dtype_name = jnp.dtype(dtype).name
    print("Merge/split microbench")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Trials: {trials}, warmup: {warmup}, dtype: {dtype_name}")
    print("Methods:", ", ".join(methods.keys()))
    print(
        "Config:",
        f"block_size={os.environ.get('XTRUCTURE_BGPQ_MERGE_BLOCK_SIZE', 'auto')}",
        f"unroll_max={os.environ.get('XTRUCTURE_BGPQ_MERGE_UNROLL_MAX', '32')}",
        f"num_warps={os.environ.get('XTRUCTURE_BGPQ_MERGE_NUM_WARPS', 'auto')}",
        f"num_stages={os.environ.get('XTRUCTURE_BGPQ_MERGE_NUM_STAGES', 'auto')}",
        f"value_pack={os.environ.get('XTRUCTURE_BGPQ_MERGE_VALUE_PACKING', 'auto')}",
        f"value_scalar_max={os.environ.get('XTRUCTURE_BGPQ_MERGE_VALUE_SCALAR_MAX', '16')}",
        f"value_reorder={os.environ.get('XTRUCTURE_BGPQ_VALUE_REORDER', 'gather')}",
        f"size_offset={size_offset}",
    )

    winners: Dict[int, str] = {}
    for size in sizes:
        size_b = size + size_offset
        if size_b <= 0:
            raise ValueError("size_offset produces non-positive size_b.")
        _print_size_header(size, size_b)
        key = jax.random.PRNGKey(seed + size + size_b)
        ak, bk = _make_sorted_arrays(key, size, size_b, dtype)
        total = int(ak.size + bk.size)
        parallel_config = _merge_parallel_config(total)
        print(
            "  parallel config:",
            f"block_size={parallel_config['block_size']}",
            f"unroll_max={parallel_config['unroll_max']}",
            f"num_warps={parallel_config['num_warps']}",
            f"num_stages={parallel_config['num_stages']}",
        )

        reference = merge_sort_split_idx(ak, bk)
        _block_until_ready(reference)

        active_methods: Dict[str, MethodFn] = {}
        for name, method in methods.items():
            try:
                outputs = method(ak, bk)
                _block_until_ready(outputs)
            except Exception as exc:
                print(f"  skip {name}: {type(exc).__name__}: {exc}")
                continue
            if verify:
                _verify_outputs(name, outputs, ak, bk, reference)
            active_methods[name] = method

        if not active_methods:
            raise RuntimeError("All benchmark methods failed; nothing to benchmark.")

        if verify:
            print("  verification: ok")

        for _ in range(warmup):
            for method in active_methods.values():
                _block_until_ready(method(ak, bk))

        results: Dict[str, Dict[str, float]] = {}
        throughput: Dict[str, float] = {}
        for name, method in active_methods.items():
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


def _make_merge_sort_split_fn(backend: MethodFn) -> ValueMethodFn:
    def _gather_sorted_values(av: Xtructurable, bv: Xtructurable, sorted_idx: chex.Array):
        reorder_mode = os.environ.get("XTRUCTURE_BGPQ_VALUE_REORDER", "gather").strip().lower()
        if reorder_mode in {"concat", "concat_gather"}:
            val = xnp.concatenate([av, bv], axis=0)
            return val[sorted_idx]
        if reorder_mode not in {"gather", "direct"}:
            raise ValueError("Invalid XTRUCTURE_BGPQ_VALUE_REORDER. Expected gather or concat.")
        n = jax.tree_util.tree_leaves(av)[0].shape[0]
        m = jax.tree_util.tree_leaves(bv)[0].shape[0]
        if n == 0:
            return bv[sorted_idx]
        if m == 0:
            return av[sorted_idx]
        idx = jnp.asarray(sorted_idx, dtype=jnp.int32)
        idx_a = jnp.clip(idx, 0, n - 1)
        idx_b = jnp.clip(idx - n, 0, m - 1)
        take_a = idx < n
        val_a = av[idx_a]
        val_b = bv[idx_b]

        def _select_leaf(leaf_a, leaf_b):
            cond = take_a
            if leaf_a.ndim > 1:
                cond = cond.reshape((cond.shape[0],) + (1,) * (leaf_a.ndim - 1))
            return jnp.where(cond, leaf_a, leaf_b)

        return jax.tree_util.tree_map(_select_leaf, val_a, val_b)

    @jax.jit
    def _fn(ak: chex.Array, av: Xtructurable, bk: chex.Array, bv: Xtructurable):
        n = int(ak.size)
        sorted_key, sorted_idx = backend(ak, bk)
        sorted_key_any = cast(Any, sorted_key)
        sorted_val = cast(Any, _gather_sorted_values(av, bv, sorted_idx))
        return (
            sorted_key_any[:n],
            sorted_val[:n],
            sorted_key_any[n:],
            sorted_val[n:],
        )

    return _fn


def _make_merge_sort_split_kv_fn(backend) -> ValueMethodFn:
    @jax.jit
    def _fn(ak: chex.Array, av: Xtructurable, bk: chex.Array, bv: Xtructurable):
        n = int(ak.size)
        sorted_key, sorted_val = backend(ak, av, bk, bv)
        sorted_key_any = cast(Any, sorted_key)
        sorted_val_any = cast(Any, sorted_val)
        return (
            sorted_key_any[:n],
            sorted_val_any[:n],
            sorted_key_any[n:],
            sorted_val_any[n:],
        )

    return _fn


def run_bench_values(
    sizes: List[int],
    trials: int,
    warmup: int,
    dtype: jnp.dtype,
    methods: Dict[str, MethodFn],
    seed: int,
    verify: bool,
    value_cls,
    size_offset: int,
) -> None:
    dtype_name = jnp.dtype(dtype).name
    print("\nMerge/split microbench (keys+values)")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Trials: {trials}, warmup: {warmup}, dtype: {dtype_name}")
    print("Methods:", ", ".join(methods.keys()))
    print(
        "Config:",
        f"block_size={os.environ.get('XTRUCTURE_BGPQ_MERGE_BLOCK_SIZE', 'auto')}",
        f"unroll_max={os.environ.get('XTRUCTURE_BGPQ_MERGE_UNROLL_MAX', '32')}",
        f"num_warps={os.environ.get('XTRUCTURE_BGPQ_MERGE_NUM_WARPS', 'auto')}",
        f"num_stages={os.environ.get('XTRUCTURE_BGPQ_MERGE_NUM_STAGES', 'auto')}",
        f"value_pack={os.environ.get('XTRUCTURE_BGPQ_MERGE_VALUE_PACKING', 'auto')}",
        f"value_scalar_max={os.environ.get('XTRUCTURE_BGPQ_MERGE_VALUE_SCALAR_MAX', '16')}",
        f"value_reorder={os.environ.get('XTRUCTURE_BGPQ_VALUE_REORDER', 'gather')}",
        f"size_offset={size_offset}",
    )

    reference_method = _make_merge_sort_split_fn(merge_sort_split_idx)
    value_methods: Dict[str, ValueMethodFn] = {
        name: _make_merge_sort_split_fn(method) for name, method in methods.items()
    }
    value_methods["parallel_kv"] = _make_merge_sort_split_kv_fn(merge_arrays_parallel_kv)
    print("Value methods:", ", ".join(value_methods.keys()))

    winners: Dict[int, str] = {}
    for size in sizes:
        size_b = size + size_offset
        if size_b <= 0:
            raise ValueError("size_offset produces non-positive size_b.")
        _print_size_header(size, size_b)
        key = jax.random.PRNGKey(seed + size + size_b)
        ak, bk, av, bv = _make_sorted_inputs(key, size, size_b, dtype, value_cls)
        total = int(ak.size + bk.size)
        parallel_config = _merge_parallel_config(total)
        print(
            "  parallel config:",
            f"block_size={parallel_config['block_size']}",
            f"unroll_max={parallel_config['unroll_max']}",
            f"num_warps={parallel_config['num_warps']}",
            f"num_stages={parallel_config['num_stages']}",
        )

        reference = reference_method(ak, av, bk, bv)
        _block_until_ready(reference)

        active_methods: Dict[str, ValueMethodFn] = {}
        for name, method in value_methods.items():
            try:
                outputs = method(ak, av, bk, bv)
                _block_until_ready(outputs)
            except Exception as exc:
                print(f"  skip {name}: {type(exc).__name__}: {exc}")
                continue
            if verify:
                _verify_value_outputs(name, outputs, reference)
            active_methods[name] = method

        if not active_methods:
            raise RuntimeError("All benchmark methods failed; nothing to benchmark.")

        if verify:
            print("  verification: ok")

        for _ in range(warmup):
            for method in active_methods.values():
                _block_until_ready(method(ak, av, bk, bv))

        results: Dict[str, Dict[str, float]] = {}
        throughput: Dict[str, float] = {}
        for name, method in active_methods.items():
            durations = _time_method_values(method, ak, av, bk, bv, trials)
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
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate outputs against the reference method before timing.",
    )
    parser.add_argument(
        "--sweep-triton",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sweep Triton num_warps/num_stages combinations.",
    )
    parser.add_argument(
        "--warps",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Triton num_warps values to sweep (requires --sweep-triton).",
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Triton num_stages values to sweep (requires --sweep-triton).",
    )
    parser.add_argument(
        "--size-offset",
        type=int,
        default=0,
        help="Offset for second array size (m = n + offset).",
    )
    parser.add_argument(
        "--bench-values",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Benchmark full merge_sort_split (keys + values).",
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

    if args.sweep_triton:
        base_warps = os.environ.get("XTRUCTURE_BGPQ_MERGE_NUM_WARPS")
        base_stages = os.environ.get("XTRUCTURE_BGPQ_MERGE_NUM_STAGES")
        try:
            for num_warps, num_stages in itertools.product(args.warps, args.stages):
                os.environ["XTRUCTURE_BGPQ_MERGE_NUM_WARPS"] = str(num_warps)
                os.environ["XTRUCTURE_BGPQ_MERGE_NUM_STAGES"] = str(num_stages)
                print(f"\n=== Triton sweep num_warps={num_warps} num_stages={num_stages} ===")
                run_bench(
                    sizes=args.sizes,
                    trials=args.trials,
                    warmup=args.warmup,
                    dtype=dtype,
                    methods=methods,
                    seed=args.seed,
                    verify=args.verify,
                    size_offset=args.size_offset,
                )
                if args.bench_values:
                    run_bench_values(
                        sizes=args.sizes,
                        trials=args.trials,
                        warmup=args.warmup,
                        dtype=dtype,
                        methods=methods,
                        seed=args.seed,
                        verify=args.verify,
                        value_cls=BenchValue,
                        size_offset=args.size_offset,
                    )
        finally:
            if base_warps is None:
                os.environ.pop("XTRUCTURE_BGPQ_MERGE_NUM_WARPS", None)
            else:
                os.environ["XTRUCTURE_BGPQ_MERGE_NUM_WARPS"] = base_warps
            if base_stages is None:
                os.environ.pop("XTRUCTURE_BGPQ_MERGE_NUM_STAGES", None)
            else:
                os.environ["XTRUCTURE_BGPQ_MERGE_NUM_STAGES"] = base_stages
        return

    run_bench(
        sizes=args.sizes,
        trials=args.trials,
        warmup=args.warmup,
        dtype=dtype,
        methods=methods,
        seed=args.seed,
        verify=args.verify,
        size_offset=args.size_offset,
    )
    if args.bench_values:
        run_bench_values(
            sizes=args.sizes,
            trials=args.trials,
            warmup=args.warmup,
            dtype=dtype,
            methods=methods,
            seed=args.seed,
            verify=args.verify,
            value_cls=BenchValue,
            size_offset=args.size_offset,
        )


if __name__ == "__main__":
    main()
