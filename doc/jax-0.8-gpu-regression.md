# JAX 0.8.0 GPU Regression Report

## Overview
- **Impacted Components:** `xtructure.hashtable.HashTable.parallel_insert` and `xtructure.bgpq.BGPQ.insert`
- **JAX Version:** 0.8.0 (GPU backend)
- **Status:** Deterministic failure on GPU; CPU backend still passes existing test suite.
- **Introduced:** After upgrading from JAX 0.7.2 to 0.8.0.

## Symptoms
- `pytest tests/hash_test.py::test_hash_table_insert` reports that every lookup returns `False` immediately after insertion despite `table.size` increasing.
- Multiple heap tests under `tests/heap_test.py` report mismatched keys and values after batched inserts.
- Failures only occur when `JAX_PLATFORM_NAME=gpu`; CPU runs continue to succeed.

## Reproduction Steps
1. Activate the GPU-enabled environment (`conda activate py312`).
2. Ensure `jax==0.8.0` and the CUDA runtime are visible.
3. Run the targeted tests:
   ```bash
   JAX_PLATFORM_NAME=gpu pytest tests/hash_test.py::test_hash_table_insert -vv
   JAX_PLATFORM_NAME=gpu pytest tests/heap_test.py -k insert_and_delete -vv
   ```
4. Observe that lookups return entirely `False` and heap invariants fail.

## Detailed Findings
- Hash table insertions update `table.size` and `table.table_idx`, but the storage buffer (`table.table`) remains filled with default sentinel values (`0xFF` and `0xFFFFFFFF`).
- The lookup path correctly computes the target indices yet retrieves sentinel values, so every `found` flag remains `False`.
- The BGPQ priority queue shows analogous corruption: `key_store` and `val_store` do not stay aligned after insertions, leading to mismatched key/value pairs that cascade into delete-min checks.
- The regressions reproduce with minimal batches (e.g., inserting two items) and do not depend on duplicate handling logic.

## Root Cause Hypothesis
- Both data structures mutate large dataclass buffers inside `lax.while_loop` bodies while carrying the entire structure as loop state.
- JAX 0.8.0 introduces a new GPU lowering path (part of the `pmap` → `jit`/`shard_map` migration) that performs more aggressive buffer aliasing for loop-carried arrays.
- Because the mutation helpers (`set_as_condition`, scatter updates) rely on the buffers behaving like fully materialized copies, aliasing causes the stores executed in one iteration to overwrite or bypass the intended slices in subsequent iterations.
- CPU execution still “works” because the XSLA CPU backend materializes distinct buffers per iteration, masking the latent bug that the GPU backend now exposes.

## Suggested Fixes
- Refactor the mutation strategy to avoid passing the whole table/heap arrays through the while loop state.
  - Option A: Build per-iteration deltas (indices + values) and apply them once outside the loop with `lax.dynamic_update_index_in_dim` or batched scatter updates.
  - Option B: Explicitly copy the table buffers at the start of the loop to break aliasing (functional but less efficient).
- Audit all usages of `set_as_condition` and custom scatter helpers to ensure they return new arrays instead of mutating aliased buffers.
- Add GPU-specific regression tests (e.g., GitHub Action worker with CUDA) to ensure future changes fail fast if aliasing reappears.

## Mitigation Options
- Short term: Pin production GPU builds to `jax<=0.7.2` until the refactor lands.
- Medium term: Provide an environment flag to route GPU insertions through a slower but functional path (e.g., CPU fallback or batched rebuild) while the optimized rewrite is in progress.

## Next Steps
1. Prototype a copy-free rewrite of `HashTable._parallel_insert` that keeps the mutation localized to immutable slices.
2. Apply analogous changes to the BGPQ merge/heapify routines.
3. Extend the test matrix with `JAX_PLATFORM_NAME=gpu` to cover the repaired paths.
4. Document the architectural change in the developer guide once the fix is merged.
