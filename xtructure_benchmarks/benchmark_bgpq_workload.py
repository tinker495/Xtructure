import argparse

import jax
import jax.numpy as jnp

from xtructure import BGPQ, FieldDescriptor, xtructure_dataclass
from xtructure_benchmarks.harness import (
    BenchmarkResult,
    add_harness_args,
    configure_precision,
    finalize_result,
    run_case,
)


@xtructure_dataclass
class SmallValue:
    x: FieldDescriptor.scalar(dtype=jnp.uint32)


@xtructure_dataclass
class BigValue:
    x: FieldDescriptor.tensor(dtype=jnp.float32, shape=(64,))


def _pop_process_mask(keys: jax.Array, pop_ratio: float, min_pop: int) -> jax.Array:
    filled = jnp.isfinite(keys)
    mult = jnp.maximum(1.0 + pop_ratio, 1.01)
    threshold = keys[0] * mult + 1e-6
    base = jnp.logical_and(filled, keys <= threshold)
    min_mask = jnp.logical_and(jnp.cumsum(filled) <= min_pop, filled)
    return jnp.logical_or(base, min_mask)


def _parse_batch_sizes(batch_sizes: str) -> list[int]:
    if not batch_sizes:
        return []
    return [int(x.strip()) for x in batch_sizes.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="BGPQ steady-state workload benchmark")
    add_harness_args(parser)
    parser.add_argument("--max-nodes", type=int, default=2**18)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        help="Comma-separated batch sizes (e.g. 1024,4096,16384). Overrides --batch-size.",
    )
    parser.add_argument("--prefill", type=int, default=32)
    parser.add_argument("--branching-factor", type=int, default=2)
    parser.add_argument("--min-pop", type=int, default=1)
    parser.add_argument("--pop-ratio", type=float, default=0.5)
    parser.add_argument(
        "--pop-calls",
        type=int,
        default=1,
        help="Number of delete_mins() calls per step (default: 1).",
    )
    parser.add_argument(
        "--value-kind", choices=["u32_small", "big_payload"], default="u32_small"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="xtructure_benchmarks/results/bgpq_workload_results.json",
    )
    args = parser.parse_args()
    configure_precision(args)

    batch_sizes = _parse_batch_sizes(args.batch_sizes) or [args.batch_size]
    value_cls = SmallValue if args.value_kind == "u32_small" else BigValue
    result = BenchmarkResult()

    base_key = jax.random.PRNGKey(args.seed)

    for batch_size in batch_sizes:
        heap = BGPQ.build(args.max_nodes, batch_size, value_cls, jnp.float32)

        # Derive per-batch-size RNG so multi-size runs stay reproducible.
        k0 = jax.random.fold_in(base_key, int(batch_size))
        k_prefill_keys, k_prefill_vals, run_key = jax.random.split(k0, 3)
        pre_keys = jax.random.uniform(
            k_prefill_keys, (args.prefill, batch_size), dtype=jnp.float32
        )
        pre_vals = value_cls.random(
            shape=(args.prefill, batch_size), key=k_prefill_vals
        )

        @jax.jit
        def prefill(h, keys, vals):
            def body(i, carry):
                return carry.insert(
                    keys[i], jax.tree_util.tree_map(lambda x: x[i], vals)
                )

            return jax.lax.fori_loop(0, keys.shape[0], body, h)

        heap = prefill(heap, pre_keys, pre_vals)
        jax.block_until_ready(heap)

        @jax.jit
        def one_step(state):
            h, k = state
            k, sk, sv = jax.random.split(k, 3)
            child_keys = jax.random.uniform(
                sk,
                (args.pop_calls, args.branching_factor, batch_size),
                dtype=jnp.float32,
            )
            child_vals = value_cls.random(
                shape=(args.pop_calls, args.branching_factor, batch_size), key=sv
            )

            processed = jnp.asarray(0, dtype=jnp.int32)

            def pop_body(i, carry):
                hp, proc = carry
                hp, popped_keys, popped_vals = hp.delete_mins()
                process_mask = _pop_process_mask(
                    popped_keys, args.pop_ratio, args.min_pop
                )
                requeue_keys = jnp.where(process_mask, jnp.inf, popped_keys)
                hp = hp.insert(requeue_keys, popped_vals)

                row_child_keys = child_keys[i]
                row_child_vals = jax.tree_util.tree_map(lambda x: x[i], child_vals)

                def insert_child_row(j, hcarry):
                    masked_keys = jnp.where(process_mask, row_child_keys[j], jnp.inf)
                    row_vals = jax.tree_util.tree_map(lambda x: x[j], row_child_vals)
                    return hcarry.insert(masked_keys, row_vals)

                hp = jax.lax.fori_loop(0, args.branching_factor, insert_child_row, hp)
                return hp, proc + jnp.sum(process_mask, dtype=jnp.int32)

            h, processed = jax.lax.fori_loop(
                0, args.pop_calls, pop_body, (h, processed)
            )
            return (h, k), processed

        @jax.jit
        def run_inner(state):
            def body(_, carry):
                core, total_processed = carry
                core, processed = one_step(core)
                return core, total_processed + processed

            return jax.lax.fori_loop(
                0, args.inner_steps, body, (state, jnp.asarray(0, jnp.int32))
            )

        def fn():
            (h, _), processed = run_inner((heap, run_key))
            if args.transfer_policy == "payload_only":
                return processed
            return (h.heap_size, h.buffer_size, processed)

        candidates_per_call = (
            batch_size * args.branching_factor * args.pop_calls * args.inner_steps
        )
        run_case(
            result,
            name="bgpq_frontier_step",
            params={
                "max_nodes": args.max_nodes,
                "batch_size": int(batch_size),
                "prefill": args.prefill,
                "branching_factor": args.branching_factor,
                "min_pop": args.min_pop,
                "pop_ratio": args.pop_ratio,
                "pop_calls": args.pop_calls,
                "value_kind": args.value_kind,
            },
            payload_items=int(candidates_per_call),
            fn=fn,
            args=args,
        )

    finalize_result(result, args, args.output, extra_run={"batch_sizes": batch_sizes})


if __name__ == "__main__":
    main()
