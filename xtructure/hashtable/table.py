"""HashTable data container and public API."""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable, base_dataclass
from .constants import SIZE_DTYPE
from .insert import _hashtable_insert_jit, _hashtable_parallel_insert_jit
from .lookup import (
    _hashtable_getitem_jit,
    _hashtable_lookup_bucket_jit,
    _hashtable_lookup_jit,
    _hashtable_lookup_parallel_jit,
)


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
    bucket_fill_levels = jnp.zeros((_capacity + 1), dtype=SIZE_DTYPE)
    bucket_occupancy = jnp.zeros((_capacity + 1), dtype=jnp.uint32)
    fingerprints = jnp.zeros(((_capacity + 1) * bucket_size,), dtype=jnp.uint32)
    table_cls = cast(Any, HashTable)
    return table_cls(
        seed,
        capacity,
        _capacity,
        bucket_size,
        size,
        table,
        bucket_fill_levels,
        bucket_occupancy,
        fingerprints,
        int(max_probes),
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
    bucket_occupancy: chex.Array
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

    def lookup_bucket(self, input: Xtructurable) -> tuple[Xtructurable, chex.Array, chex.Array]:
        return _hashtable_lookup_bucket_jit(self, input)

    def lookup(self, input: Xtructurable) -> tuple[Xtructurable, bool]:
        return _hashtable_lookup_jit(self, input)

    def lookup_parallel(
        self, inputs: Xtructurable, filled: chex.Array | bool = True
    ) -> tuple[Xtructurable, chex.Array]:
        return _hashtable_lookup_parallel_jit(self, inputs, filled)

    def insert(self, input: Xtructurable) -> tuple["HashTable", bool, Xtructurable]:
        return _hashtable_insert_jit(self, input)

    def parallel_insert(
        self,
        inputs: Xtructurable,
        filled: chex.Array | bool | None = None,
        unique_key: chex.Array | None = None,
    ):
        return _hashtable_parallel_insert_jit(self, inputs, filled, unique_key)

    def __getitem__(self, idx: Xtructurable) -> Xtructurable:
        return _hashtable_getitem_jit(self, idx)
