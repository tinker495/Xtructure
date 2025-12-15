# API Conventions and Migration Notes

This page collects the user-facing API rules that the project will follow going
forward so that you can learn Xtructure once and keep using it consistently.

## Imports

- Start every project with `import xtructure as xt`.
- Use `xt.xnp` for NumPy-like helpers. The previous aliases
  (`xtructure.numpy`, `xtructure.xtructure_numpy`) remain for backward
  compatibility but now emit deprecation warnings from `xtructure.numpy`.
- Public symbols are exported from `xtructure.__all__`; subpackages are
  considered internal unless referenced here or in the README.

## Object-first style

All data structures should be used through their instance methods. The
function-style helpers remain for compatibility, but documentation and examples
use methods exclusively. For example:

```python
pq = xt.BGPQ.build(max_size=2000, batch_size=64, value_class=MyItem)
pq = pq.insert(keys, values)
pq, keys, values = pq.delete_mins()

ht = xt.HashTable.build(MyItem, n_batches=1, capacity=1024)
ht, inserted_mask, unique_mask, idxs = ht.parallel_insert(batch)
idxs, found = ht.lookup_parallel(batch)
```

## Batched inputs and masks

Operations that accept batched inputs will accept optional boolean masks
(`mask`/`filled`) to avoid manual padding loops. Return values include masks
where needed (for example, deletion operations expose `valid_mask` for the
actual number of items removed). This keeps variable-length batches predictable
without sacrificing safety.

## Stable vs. experimental surface

Only the symbols re-exported from `xtructure.__all__` are considered stable. If
an API lives under an internal module (for example, `xtructure.core`) and is not
re-exported, treat it as experimental and subject to change.
