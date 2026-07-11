"""HashTable data container and public API."""

from __future__ import annotations

from functools import partial

import chex
import jax
import jax.numpy as jnp

from ..core.dataclass import base_dataclass
from ..core.dtype_facts import SIZE_DTYPE
from ..core.protocol import Xtructurable
from .insert import _hashtable_insert_jit, _hashtable_parallel_insert_jit
from .lookup import (
    _hashtable_getitem_jit,
    _hashtable_lookup_bucket_jit,
    _hashtable_lookup_jit,
    _hashtable_lookup_parallel_with_probe_jit,
    _lookup_parallel_dispatch,
)
from .types import BucketIdx, HashIdx, HashTableProbe


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
def _hashtable_build_jit(
    dataclass: Xtructurable,
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
        dataclass: Xtructurable,
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

    def lookup_bucket(self, input: Xtructurable) -> tuple[BucketIdx, bool, chex.Array]:
        return _hashtable_lookup_bucket_jit(self, input)

    def lookup(self, input: Xtructurable) -> tuple[HashIdx, bool]:
        return _hashtable_lookup_jit(self, input)

    def lookup_parallel(
        self, inputs: Xtructurable, filled: chex.Array | bool = True
    ) -> tuple[HashIdx, chex.Array]:
        return _lookup_parallel_dispatch(self, inputs, filled)

    def lookup_parallel_with_probe(
        self, inputs: Xtructurable, filled: chex.Array | bool = True
    ) -> tuple[HashIdx, chex.Array, HashTableProbe]:
        """Parallel lookup that also returns the shared :class:`HashTableProbe`.

        The probe carries the products of the single uint32 hash pass this lookup
        already computed. A caller that inserts the same batch of states next can
        pass it back to :meth:`parallel_insert` (``probe=...``) to skip the second,
        redundant hash pass. Table behaviour is identical to :meth:`lookup_parallel`.
        """
        return _hashtable_lookup_parallel_with_probe_jit(self, inputs, filled)

    def insert(self, input: Xtructurable) -> tuple["HashTable", bool, HashIdx]:
        return _hashtable_insert_jit(self, input)

    def parallel_insert(
        self,
        inputs: Xtructurable,
        filled: chex.Array | bool = None,
        unique_key: chex.Array = None,
        probe: HashTableProbe | None = None,
    ):
        """Insert a batch of states in parallel.

        When ``probe`` is supplied it must be the :class:`HashTableProbe` returned
        by :meth:`lookup_parallel_with_probe` for this exact batch and table; the
        insert then reuses that hash pass instead of recomputing it, producing
        bit-identical table state. Shape/dtype mismatches raise (no silent recompute).
        """
        return _hashtable_parallel_insert_jit(self, inputs, filled, unique_key, probe)

    def __getitem__(self, idx: HashIdx) -> Xtructurable:
        return _hashtable_getitem_jit(self, idx)
