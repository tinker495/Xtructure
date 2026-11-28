"""HashTable data container and public API."""

from __future__ import annotations

from functools import partial

import chex
import jax
import jax.numpy as jnp

from ..core.dataclass import base_dataclass
from ..core.dtype_facts import SIZE_DTYPE
from ..core.protocol import Xtructurable
from .insert import (
    _hashtable_insert_jit,
    _hashtable_parallel_insert_jit,
    _parallel_insert_resolved_jit,
)
from .lookup import (
    _hashtable_getitem_jit,
    _hashtable_lookup_bucket_jit,
    _hashtable_lookup_jit,
    _lookup_parallel_dispatch,
    _lookup_parallel_with_probe_dispatch,
    _resolve_parallel_jit,
)
from .types import BucketIdx, HashIdx, HashTableProbe, ResolvedHashSlots


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
def _hashtable_build_jit(
    dataclass: type[Xtructurable],
    seed: int,
    capacity: int,
    bucket_size: int = 2,
    hash_size_multiplier: int = 2,
    max_probes: int | None = None,
) -> "HashTable":
    _target_cap = int(hash_size_multiplier * capacity / bucket_size)
    if _target_cap <= 1:
        _capacity = 1
    else:
        _capacity = 1 << (_target_cap - 1).bit_length()

    size = SIZE_DTYPE(0)
    if max_probes is None:
        max_probes = _capacity * bucket_size

    table = dataclass.default(((_capacity + 1) * bucket_size,))
    bucket_fill_levels = jnp.zeros((_capacity + 1), dtype=jnp.uint8)
    fingerprints = jnp.zeros(((_capacity + 1) * bucket_size,), dtype=jnp.uint32)
    return HashTable(
        seed=seed,
        capacity=capacity,
        _capacity=_capacity,
        bucket_size=bucket_size,
        size=size,
        table=table,
        bucket_fill_levels=bucket_fill_levels,
        fingerprints=fingerprints,
        max_probes=int(max_probes),
    )


@base_dataclass(
    frozen=True,
    static_fields=("seed", "capacity", "_capacity", "bucket_size", "max_probes"),
)
class HashTable:
    """
    Bucketed Double Hash Table Implementation

    Uses double hashing with buckets to resolve collisions.
    """

    seed: int
    capacity: int
    _capacity: int
    bucket_size: int
    size: int
    table: Xtructurable
    bucket_fill_levels: chex.Array
    fingerprints: chex.Array
    max_probes: int

    @staticmethod
    def build(
        dataclass: type[Xtructurable],
        seed: int,
        capacity: int,
        bucket_size: int = 8,
        hash_size_multiplier: int = 2,
        max_probes: int | None = None,
    ) -> "HashTable":
        """
        Initialize a new hash table backed by JAX-friendly storage.
        """
        return _hashtable_build_jit(
            dataclass, seed, capacity, bucket_size, hash_size_multiplier, max_probes
        )

    def lookup_bucket(self, input: Xtructurable) -> tuple[BucketIdx, jax.Array, jax.Array]:
        return _hashtable_lookup_bucket_jit(self, input)

    def lookup(self, input: Xtructurable) -> tuple[HashIdx, jax.Array]:
        return _hashtable_lookup_jit(self, input)

    def lookup_parallel(
        self,
        inputs: Xtructurable,
        filled: chex.Array | bool | None = True,
        probe: HashTableProbe | None = None,
        *,
        assume_unique: bool = False,
    ) -> tuple[HashIdx, jax.Array]:
        """Look up a batch of states in parallel with internal deduplication.

        By default (``assume_unique=False``), duplicate active states within
        ``inputs`` are deduplicated in the same uint32 hash pass so the table is
        probed at most once per distinct state, and the representative's
        ``(HashIdx, found)`` result is restored to every active duplicate row.

        Set ``assume_unique=True`` only when active entries in ``inputs`` are
        already unique to skip redundant intra-batch deduplication.
        """
        return _lookup_parallel_dispatch(
            self,
            inputs,
            filled,
            probe,
            assume_unique=assume_unique,
        )

    def lookup_parallel_with_probe(
        self,
        inputs: Xtructurable,
        filled: chex.Array | bool | None = True,
        probe: HashTableProbe | None = None,
        *,
        assume_unique: bool = False,
    ) -> tuple[HashIdx, jax.Array, HashTableProbe]:
        """Parallel lookup that also returns the shared :class:`HashTableProbe`.

        The probe carries the products of the single uint32 hash pass this lookup
        already computed. A caller that inserts the same batch of states next can
        pass it back to :meth:`parallel_insert` (``probe=...``) to skip the second,
        redundant hash pass. Table behaviour is identical to :meth:`lookup_parallel`.
        """
        return _lookup_parallel_with_probe_dispatch(
            self,
            inputs,
            filled,
            probe,
            assume_unique=assume_unique,
        )

    def resolve_parallel(
        self, inputs: Xtructurable, filled: jax.Array
    ) -> tuple[HashIdx, jax.Array, ResolvedHashSlots]:
        """Look up active rows and retain the resolved slots for a later insertion.

        Both inputs and their boolean mask have one batch axis. The first two
        results match lookup_parallel(..., assume_unique=True).
        """
        return _resolve_parallel_jit(self, inputs, filled)

    def parallel_insert_resolved(
        self, inputs: Xtructurable, slots: ResolvedHashSlots, filled: jax.Array
    ) -> tuple["HashTable", jax.Array, jax.Array, HashIdx]:
        """Insert unique selected rows using slots from resolve_parallel on this snapshot.

        Rows may be reordered, sliced, or combined from lookups on the same
        unchanged table. Move all slot leaves together with their input rows.
        Every selected row must have been active when resolved. The result
        follows parallel_insert(..., assume_unique=True), including masked lanes.
        """
        return _parallel_insert_resolved_jit(self, inputs, slots, filled)

    def insert(self, input: Xtructurable) -> tuple["HashTable", jax.Array, HashIdx]:
        return _hashtable_insert_jit(self, input)

    def parallel_insert(
        self,
        inputs: Xtructurable,
        filled: chex.Array | bool | None = None,
        unique_key: chex.Array | None = None,
        probe: HashTableProbe | None = None,
        *,
        assume_unique: bool = False,
    ):
        """Insert a batch of states in parallel.

        When ``probe`` is supplied it must be the :class:`HashTableProbe` returned
        by :meth:`lookup_parallel_with_probe` for this exact batch and table; the
        insert then reuses that hash pass instead of recomputing it, producing
        bit-identical table state. Shape/dtype mismatches raise (no silent recompute).

        Set ``assume_unique=True`` only when every row selected by ``filled`` is
        already unique. This skips the otherwise redundant intra-batch deduplication.
        """
        return _hashtable_parallel_insert_jit(
            self,
            inputs,
            filled,
            unique_key,
            probe,
            assume_unique,
        )

    def __getitem__(self, idx: HashIdx) -> Xtructurable:
        return _hashtable_getitem_jit(self, idx)
