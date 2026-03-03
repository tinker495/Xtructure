import argparse

import jax
import jax.numpy as jnp

from xtructure import FieldDescriptor, xtructure_dataclass
from xtructure import numpy as xnp
from xtructure_benchmarks.harness import (
    BenchmarkResult,
    add_harness_args,
    configure_precision,
    finalize_result,
    run_case,
)


@xtructure_dataclass
class NodeState:
    key: FieldDescriptor.scalar(dtype=jnp.uint32)
    cost: FieldDescriptor.scalar(dtype=jnp.float32)
    parent: FieldDescriptor.scalar(dtype=jnp.uint32)


def _parse_batch_sizes(batch_sizes: str) -> list[int]:
    if not batch_sizes:
        return []
    return [int(x.strip()) for x in batch_sizes.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="xnp pipeline benchmark")
    add_harness_args(parser)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        help="Comma-separated batch sizes (e.g. 1024,4096,16384). Overrides --batch-size.",
    )
    parser.add_argument("--dup-ratio", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=str,
        default="xtructure_benchmarks/results/xnp_pipeline_results.json",
    )
    args = parser.parse_args()
    configure_precision(args)

    result = BenchmarkResult()
    batch_sizes = _parse_batch_sizes(args.batch_sizes) or [args.batch_size]
    base_key = jax.random.PRNGKey(args.seed)

    for batch_size in batch_sizes:
        key = jax.random.fold_in(base_key, int(batch_size))

        @jax.jit
        def one_step(k):
            k, k1, k2 = jax.random.split(k, 3)
            nodes = NodeState.random(shape=(batch_size,), key=k1)
            dup_mask = jax.random.uniform(k2, (batch_size,)) < args.dup_ratio
            dup_nodes = jax.tree_util.tree_map(lambda x: jnp.roll(x, 1), nodes)
            nodes = jax.tree_util.tree_map(
                lambda a, b: jnp.where(dup_mask, b, a), nodes, dup_nodes
            )

            uniq_key_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
            uniq = xnp.unique_mask(nodes, key=nodes.key.astype(uniq_key_dtype))
            sort_keys = jnp.where(uniq, nodes.cost, jnp.inf)
            sorted_cost, sort_idx = jax.lax.sort_key_val(
                sort_keys, jnp.arange(batch_size)
            )
            cond = sorted_cost < jnp.inf
            updates = xnp.take(nodes, sort_idx, axis=0)
            updates = NodeState(
                key=updates.key,
                cost=updates.cost + 1.0,
                parent=updates.parent,
            )
            out = xnp.update_on_condition(nodes, sort_idx, cond, updates)
            checksum = jnp.sum(out.cost)
            return k, checksum

        @jax.jit
        def run_inner(k):
            def body(_, carry):
                nk, acc = carry
                nk, chk = one_step(nk)
                return (nk, acc + chk)

            return jax.lax.fori_loop(
                0, args.inner_steps, body, (k, jnp.asarray(0.0, jnp.float32))
            )

        def fn():
            _, checksum = run_inner(key)
            return checksum

        items_per_call = batch_size * args.inner_steps
        run_case(
            result,
            name="xnp_pipeline",
            params={"batch_size": int(batch_size), "dup_ratio": args.dup_ratio},
            payload_items=int(items_per_call),
            fn=fn,
            args=args,
        )

    finalize_result(result, args, args.output, extra_run={"batch_sizes": batch_sizes})


if __name__ == "__main__":
    main()
