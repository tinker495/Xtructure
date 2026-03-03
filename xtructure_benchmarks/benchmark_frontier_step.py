import argparse

import jax
import jax.numpy as jnp

from xtructure import (
    BGPQ,
    FieldDescriptor,
    HashTable,
    xtructure_dataclass,
)
from xtructure import (
    numpy as xnp,
)
from xtructure_benchmarks.harness import (
    BenchmarkResult,
    add_harness_args,
    configure_precision,
    finalize_result,
    run_case,
)


@xtructure_dataclass
class FrontierState:
    node_id: FieldDescriptor.scalar(dtype=jnp.uint32)


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
    parser = argparse.ArgumentParser(description="Composite frontier-step benchmark")
    add_harness_args(parser)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        help="Comma-separated batch sizes (e.g. 1024,4096,16384). Overrides --batch-size.",
    )
    parser.add_argument("--branching-factor", type=int, default=2)
    parser.add_argument("--dup-ratio", type=float, default=0.3)
    parser.add_argument("--hit-ratio", type=float, default=0.4)
    parser.add_argument("--pop-ratio", type=float, default=0.5)
    parser.add_argument("--min-pop", type=int, default=1)
    parser.add_argument("--occupancy", type=float, default=0.7)
    parser.add_argument("--capacity", type=int, default=2**16)
    parser.add_argument(
        "--output",
        type=str,
        default="xtructure_benchmarks/results/frontier_step_results.json",
    )
    args = parser.parse_args()
    configure_precision(args)

    batch_sizes = _parse_batch_sizes(args.batch_sizes) or [args.batch_size]
    prefill_n = int(args.capacity * args.occupancy)
    result = BenchmarkResult()

    base_key = jax.random.PRNGKey(args.seed)

    for batch_size in batch_sizes:
        table = HashTable.build(FrontierState, 0, args.capacity)
        heap = BGPQ.build(args.capacity, batch_size, FrontierState, jnp.float32)

        # Per-batch-size RNG (reproducible across multi-size runs).
        k0 = jax.random.fold_in(base_key, int(batch_size))
        (
            k_prefill_state,
            k_prefill_cost,
            k_init_cost,
            k_init_state,
            run_key,
        ) = jax.random.split(k0, 5)

        # Prefill table and initialize cost_table with finite costs to model hit-path rejects.
        prefill = FrontierState.random(shape=(prefill_n,), key=k_prefill_state)
        prefill_cost = jax.random.uniform(
            k_prefill_cost, (prefill_n,), dtype=jnp.float32
        )
        cost_table = jnp.full(table.table.shape.batch, jnp.inf, dtype=jnp.float32)

        table, inserted0, uniq0, hash_idx0 = table.parallel_insert(
            prefill, unique_key=prefill_cost
        )
        prefill_update = jnp.logical_and(inserted0, uniq0)
        cost_table = xnp.update_on_condition(
            cost_table, hash_idx0.index, prefill_update, prefill_cost
        )

        # Seed heap (frontier).
        initial_cost = jax.random.uniform(k_init_cost, (batch_size,), dtype=jnp.float32)
        initial_state = FrontierState.random(shape=(batch_size,), key=k_init_state)
        heap = heap.insert(initial_cost, initial_state)

        # Optional: register initial frontier costs in the cost table for consistency.
        table, inserted1, uniq1, hash_idx1 = table.parallel_insert(
            initial_state, unique_key=initial_cost
        )
        best_cost_init = cost_table[hash_idx1.index]
        init_opt = initial_cost < best_cost_init
        init_update = jnp.logical_and(uniq1, init_opt)
        cost_table = xnp.update_on_condition(
            cost_table, hash_idx1.index, init_update, initial_cost
        )

        if prefill_n > 0:
            first_valid = jnp.argmax(prefill_update)
            hit_indices = jnp.nonzero(
                prefill_update, size=prefill_n, fill_value=first_valid
            )[0]
            hit_pool = jax.tree_util.tree_map(lambda x: x[hit_indices], prefill)
            hit_enabled = jnp.any(prefill_update)
        else:
            hit_pool = jax.tree_util.tree_map(lambda x: x[:1], initial_state)
            hit_enabled = jnp.asarray(False, dtype=jnp.bool_)

        jax.block_until_ready((table, heap, cost_table, hit_pool, hit_enabled))

        @jax.jit
        def step(state):
            t, h, costs, pref_pool, hit_ok, k = state
            k, k1, k2, k3, k4, k5, k6 = jax.random.split(k, 7)
            n = batch_size * args.branching_factor

            h, popped_keys, popped_vals = h.delete_mins()
            process_mask = _pop_process_mask(popped_keys, args.pop_ratio, args.min_pop)
            requeue_keys = jnp.where(process_mask, jnp.inf, popped_keys)
            h = h.insert(requeue_keys, popped_vals)

            expand_mask_2d = jnp.broadcast_to(
                process_mask[None, :], (args.branching_factor, batch_size)
            )
            expand_mask = expand_mask_2d.reshape(n)

            cand = FrontierState.random(shape=(n,), key=k1)
            cand_cost = jax.random.uniform(k2, (n,), dtype=jnp.float32)

            dup_mask = jnp.logical_and(
                jax.random.uniform(k3, (n,)) < args.dup_ratio, expand_mask
            )
            dup_src = jax.random.randint(k4, (n,), 0, n)
            cand = jax.tree_util.tree_map(
                lambda x: jnp.where(dup_mask, x[dup_src], x), cand
            )

            hit_mask = jnp.logical_and(
                jax.random.uniform(k5, (n,)) < args.hit_ratio, hit_ok
            )
            hit_mask = jnp.logical_and(hit_mask, expand_mask)
            hit_idx = jax.random.randint(k6, (n,), 0, pref_pool.node_id.shape[0])
            hit_state = jax.tree_util.tree_map(lambda x: x[hit_idx], pref_pool)
            cand = jax.tree_util.tree_map(
                lambda a, b: jnp.where(hit_mask, b, a), cand, hit_state
            )

            t, inserted, unique_mask, hash_idx = t.parallel_insert(
                cand, filled=expand_mask, unique_key=cand_cost
            )
            best_cost = costs[hash_idx.index]
            optimal_mask = cand_cost < best_cost
            final_mask = jnp.logical_and(unique_mask, optimal_mask)
            costs = xnp.update_on_condition(
                costs, hash_idx.index, final_mask, cand_cost
            )

            cand_cost_2d = cand_cost.reshape(args.branching_factor, batch_size)
            final_mask_2d = final_mask.reshape(args.branching_factor, batch_size)
            cand_2d = jax.tree_util.tree_map(
                lambda x: x.reshape(args.branching_factor, batch_size), cand
            )

            def insert_row(carry_h, row):
                row_keys, row_mask, row_vals = row
                masked_keys = jnp.where(row_mask, row_keys, jnp.inf)
                row_has_any = jnp.any(row_mask)
                next_h = jax.lax.cond(
                    row_has_any,
                    lambda hh: hh.insert(masked_keys, row_vals),
                    lambda hh: hh,
                    carry_h,
                )
                return next_h, None

            h, _ = jax.lax.scan(insert_row, h, (cand_cost_2d, final_mask_2d, cand_2d))

            return (
                (
                    t,
                    h,
                    costs,
                    pref_pool,
                    hit_ok,
                    k,
                ),
                inserted,
                jnp.sum(process_mask, dtype=jnp.int32),
            )

        @jax.jit
        def run_inner(state):
            def body(_, carry):
                core, inserted_sum, processed_sum = carry
                core, inserted, processed = step(core)
                inserted_count = jnp.sum(inserted, dtype=jnp.int32)
                return (
                    core,
                    inserted_sum + inserted_count,
                    processed_sum + processed,
                )

            return jax.lax.fori_loop(
                0,
                args.inner_steps,
                body,
                (state, jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32)),
            )

        def fn():
            (t, h, c, _, _, _), inserted_sum, processed_sum = run_inner(
                (table, heap, cost_table, hit_pool, hit_enabled, run_key)
            )
            if args.transfer_policy == "payload_only":
                return (inserted_sum, processed_sum)
            return (
                t.bucket_fill_levels,
                h.heap_size,
                h.buffer_size,
                inserted_sum,
                processed_sum,
                jnp.sum(jnp.isfinite(c)),
            )

        candidates_per_call = batch_size * args.branching_factor * args.inner_steps
        run_case(
            result,
            name="frontier_step",
            params={
                "batch_size": int(batch_size),
                "branching_factor": args.branching_factor,
                "dup_ratio": args.dup_ratio,
                "hit_ratio": args.hit_ratio,
                "pop_ratio": args.pop_ratio,
                "min_pop": args.min_pop,
                "occupancy": args.occupancy,
                "capacity": args.capacity,
            },
            payload_items=int(candidates_per_call),
            fn=fn,
            args=args,
        )

    finalize_result(result, args, args.output, extra_run={"batch_sizes": batch_sizes})


if __name__ == "__main__":
    main()
