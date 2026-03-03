import argparse

import jax
import jax.numpy as jnp

from xtructure import FieldDescriptor, HashTable, xtructure_dataclass
from xtructure_benchmarks.harness import (
    BenchmarkResult,
    add_harness_args,
    configure_precision,
    finalize_result,
    run_case,
)


@xtructure_dataclass
class WorkKey:
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    tag: FieldDescriptor.scalar(dtype=jnp.uint32)


def _inject_dups(values: WorkKey, dup_ratio: float, key: jax.Array) -> WorkKey:
    n = values.id.shape[0]
    k_mask, k_idx = jax.random.split(key)
    dup_mask = jax.random.uniform(k_mask, (n,)) < dup_ratio
    src_idx = jax.random.randint(k_idx, (n,), 0, n)
    dup_vals = jax.tree_util.tree_map(lambda x: x[src_idx], values)
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(dup_mask, b, a), values, dup_vals
    )


def _inject_hits(
    values: WorkKey,
    hit_pool: WorkKey,
    hit_ratio: float,
    key: jax.Array,
    enabled: jax.Array,
) -> WorkKey:
    n = values.id.shape[0]
    m = hit_pool.id.shape[0]
    k_mask, k_idx = jax.random.split(key)
    hit_mask = jnp.logical_and(jax.random.uniform(k_mask, (n,)) < hit_ratio, enabled)
    idx = jax.random.randint(k_idx, (n,), 0, m)
    hit_vals = jax.tree_util.tree_map(lambda x: x[idx], hit_pool)
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(hit_mask, b, a), values, hit_vals
    )


def _parse_batch_sizes(batch_sizes: str) -> list[int]:
    if not batch_sizes:
        return []
    return [int(x.strip()) for x in batch_sizes.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hashtable steady-state workload benchmark"
    )
    add_harness_args(parser)
    parser.add_argument("--capacity", type=int, default=2**16)
    parser.add_argument(
        "--max-probes",
        type=int,
        default=512,
        help="Upper bound for lookup/insert probing to keep miss-path cost bounded.",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        help="Comma-separated batch sizes (e.g. 1024,4096,16384). Overrides --batch-size.",
    )
    parser.add_argument("--prefill-occupancy", type=float, default=0.7)
    parser.add_argument("--dup-ratio", type=float, default=0.3)
    parser.add_argument("--hit-ratio", type=float, default=0.4)
    parser.add_argument("--mix-insert", type=int, default=1)
    parser.add_argument("--mix-lookup", type=int, default=1)
    parser.add_argument(
        "--output",
        type=str,
        default="xtructure_benchmarks/results/hashtable_workload_results.json",
    )
    args = parser.parse_args()
    configure_precision(args)

    prefill_n = int(args.capacity * args.prefill_occupancy)
    result = BenchmarkResult()
    batch_sizes = _parse_batch_sizes(args.batch_sizes) or [args.batch_size]

    base_key = jax.random.PRNGKey(args.seed)

    for batch_size in batch_sizes:
        table = HashTable.build(WorkKey, 0, args.capacity, max_probes=args.max_probes)
        total_slots = int(table.table.shape.batch[0])
        k0 = jax.random.fold_in(base_key, int(batch_size))
        k_prefill, run_key = jax.random.split(k0, 2)

        prefill_keys = WorkKey.random(shape=(prefill_n,), key=k_prefill)
        table, inserted0, uniq0, _ = table.parallel_insert(prefill_keys)

        if prefill_n > 0:
            prefill_inserted = jnp.logical_and(inserted0, uniq0)
            first_valid = jnp.argmax(prefill_inserted)
            hit_idx = jnp.nonzero(
                prefill_inserted, size=prefill_n, fill_value=first_valid
            )[0]
            hit_pool = jax.tree_util.tree_map(lambda x: x[hit_idx], prefill_keys)
            hit_enabled = jnp.any(prefill_inserted)
        else:
            hit_pool = WorkKey.random(shape=(1,), key=jax.random.fold_in(k_prefill, 1))
            hit_enabled = jnp.asarray(False, dtype=jnp.bool_)

        jax.block_until_ready((table, hit_pool, hit_enabled))

        @jax.jit
        def workload_step(state):
            table_state, key, hit_pool, hit_ok, lookup_hits_acc = state
            key, k_base, k_dup, k_hit = jax.random.split(key, 4)
            batch = WorkKey.random(shape=(batch_size,), key=k_base)
            batch = _inject_dups(batch, args.dup_ratio, k_dup)
            batch = _inject_hits(batch, hit_pool, args.hit_ratio, k_hit, hit_ok)

            def insert_body(_, carry):
                t = carry
                filled_now = jnp.sum(t.bucket_fill_levels, dtype=jnp.uint32)
                free_slots = jnp.maximum(
                    jnp.uint32(0), jnp.uint32(total_slots) - filled_now
                )
                insert_budget = jnp.minimum(jnp.uint32(batch_size), free_slots)
                insert_mask = jnp.arange(batch_size, dtype=jnp.uint32) < insert_budget

                def _do_insert(table_in):
                    return table_in.parallel_insert(batch, filled=insert_mask)[0]

                t = jax.lax.cond(insert_budget > 0, _do_insert, lambda x: x, t)
                return t

            def lookup_body(_, carry):
                t, lookup_hits = carry
                _, found = t.lookup_parallel(batch)
                lookup_hits = lookup_hits + jnp.sum(
                    found.astype(jnp.uint32), dtype=jnp.uint32
                )
                return t, lookup_hits

            table_next = jax.lax.fori_loop(0, args.mix_insert, insert_body, table_state)
            table_next, lookup_hits_acc = jax.lax.fori_loop(
                0,
                args.mix_lookup,
                lookup_body,
                (table_next, lookup_hits_acc),
            )
            return (table_next, key, hit_pool, hit_ok, lookup_hits_acc)

        @jax.jit
        def run_inner(state):
            return jax.lax.fori_loop(
                0, args.inner_steps, lambda _, s: workload_step(s), state
            )

        def fn():
            t, _, _, _, lookup_hits_acc = run_inner(
                (table, run_key, hit_pool, hit_enabled, jnp.uint32(0))
            )
            return t.bucket_fill_levels, lookup_hits_acc

        # Count "items" as total keys processed across the mixed operation schedule.
        items_per_step = batch_size * (args.mix_insert + args.mix_lookup)
        items_per_call = items_per_step * args.inner_steps

        run_case(
            result,
            name="hashtable_steady_state",
            params={
                "capacity": args.capacity,
                "max_probes": args.max_probes,
                "batch_size": int(batch_size),
                "prefill_occupancy": args.prefill_occupancy,
                "dup_ratio": args.dup_ratio,
                "hit_ratio": args.hit_ratio,
                "mix_insert": args.mix_insert,
                "mix_lookup": args.mix_lookup,
            },
            payload_items=int(items_per_call),
            fn=fn,
            args=args,
        )

    finalize_result(result, args, args.output, extra_run={"batch_sizes": batch_sizes})


if __name__ == "__main__":
    main()
