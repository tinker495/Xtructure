import time

import jax
import jax.numpy as jnp

from xtructure import FieldDescriptor, Stack, Queue, HashTable, xtructure_dataclass


def _block_until_ready(tree):
    """Utility: synchronously wait for all DeviceArrays inside *tree*."""
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


@xtructure_dataclass
class _Point:
    """Minimal value structure used in benchmarks."""

    x: FieldDescriptor[jnp.uint32]
    y: FieldDescriptor[jnp.uint32]


# ---------------------- Generic benchmarking helpers ---------------------- #

def _timeit(fn, *args, repeat: int = 10, warmup: int = 3):
    """Run *fn*(*args) multiple times and return the average execution time.

    A few warm-up runs are executed (and excluded from timing) to trigger JIT
    compilation.  All asynchronous JAX work is synchronised via
    _block_until_ready to produce reliable timing results.
    """
    # Warm-up (to trigger JIT compilation etc.)
    for _ in range(warmup):
        res = fn(*args)
        _block_until_ready(res)

    start = time.perf_counter()
    for _ in range(repeat):
        res = fn(*args)
        _block_until_ready(res)
    end = time.perf_counter()
    return (end - start) / repeat


# ------------------------------ Benchmarks ------------------------------ #

def benchmark_stack(batch_sizes=(1, 1024, 8192)):
    """Benchmark Stack.push / Stack.pop for several batch sizes.

    Returns a dictionary mapping batch_size -> {"push": time, "pop": time}
    (times in seconds per call).
    """
    results = {}
    for bs in batch_sizes:
        # Build an empty stack with enough head-room.
        max_size = bs * 2 + 10
        stack = Stack.build(max_size=max_size, value_class=_Point)  # type: ignore[arg-type]

        # Prepare values to push / pop.
        key = jax.random.PRNGKey(0)
        vals = _Point.random((bs,), key=key)  # type: ignore[attr-defined]

        # JIT-compile helpers.
        push_fn = jax.jit(lambda st, v: st.push(v))
        pop_fn = jax.jit(lambda st: st.pop(bs))

        # Measure push.
        push_time = _timeit(push_fn, stack, vals)

        # Push once so we have elements to pop.
        stack_full = push_fn(stack, vals)
        _block_until_ready(stack_full)

        # Measure pop (we discard returned values).
        def _pop_only(st):
            st_after, _ = pop_fn(st)
            return st_after

        pop_time = _timeit(_pop_only, stack_full)

        results[bs] = {"push": push_time, "pop": pop_time}
    return results


def benchmark_queue(batch_sizes=(1, 1024, 8192)):
    """Benchmark Queue.enqueue / Queue.dequeue for several batch sizes."""
    results = {}
    for bs in batch_sizes:
        max_size = bs * 2 + 10
        queue = Queue.build(max_size=max_size, value_class=_Point)  # type: ignore[arg-type]
        key = jax.random.PRNGKey(1)
        vals = _Point.random((bs,), key=key)  # type: ignore[attr-defined]

        enqueue_fn = jax.jit(lambda q, v: q.enqueue(v))
        dequeue_fn = jax.jit(lambda q: q.dequeue(bs))

        enqueue_time = _timeit(enqueue_fn, queue, vals)

        queue_full = enqueue_fn(queue, vals)
        _block_until_ready(queue_full)

        def _dequeue_only(q):
            q_after, _ = dequeue_fn(q)
            return q_after

        dequeue_time = _timeit(_dequeue_only, queue_full)
        results[bs] = {"enqueue": enqueue_time, "dequeue": dequeue_time}
    return results


def benchmark_hashtable(capacities=(1_000, 10_000, 50_000)):
    """Benchmark HashTable.parallel_insert / lookup for several capacities."""
    results = {}
    for cap in capacities:
        ht = HashTable.build(_Point, seed=0, capacity=cap)  # type: ignore[arg-type]

        key = jax.random.PRNGKey(2)
        items = _Point.random((cap,), key=key)  # type: ignore[attr-defined]

        # -------- insert benchmark -------- #
        def _insert_fn(table, vals):
            table_after, _, _, _ = table.parallel_insert(vals)
            return table_after

        insert_time = _timeit(_insert_fn, ht, items, repeat=3, warmup=1)

        # Perform one insertion so the table is (nearly) full for lookup.
        ht_filled = _insert_fn(ht, items)
        _block_until_ready(ht_filled)

        # Pick random element for lookup.
        sample = items[0]

        lookup_fn = jax.jit(lambda table, x: table.lookup(x))
        # Ensure compilation.
        _ = lookup_fn(ht_filled, sample)
        _block_until_ready(sample.x)  # simple device sync

        lookup_time = _timeit(lookup_fn, ht_filled, sample, repeat=100, warmup=10)

        results[cap] = {"insert_batch": insert_time, "single_lookup": lookup_time}
    return results


# ---------------------------- Entry point ----------------------------- #


def run_all_benchmarks():
    print("Running data-structure benchmarks on JAX backend:", jax.default_backend())
    print()

    stack_res = benchmark_stack()
    print("Stack (seconds per call):")
    for bs, times in stack_res.items():
        print(f"  batch_size={bs:6d}: push={times['push']*1e6:8.2f} µs | pop={times['pop']*1e6:8.2f} µs")
    print()

    queue_res = benchmark_queue()
    print("Queue (seconds per call):")
    for bs, times in queue_res.items():
        print(f"  batch_size={bs:6d}: enqueue={times['enqueue']*1e6:8.2f} µs | dequeue={times['dequeue']*1e6:8.2f} µs")
    print()

    ht_res = benchmark_hashtable()
    print("HashTable (seconds per operation):")
    for cap, times in ht_res.items():
        print(
            f"  capacity={cap:7d}: insert_batch={times['insert_batch']:.4f} s | lookup={times['single_lookup']*1e6:8.2f} µs"
        )


if __name__ == "__main__":
    run_all_benchmarks()