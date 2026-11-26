from typing import Annotated, NamedTuple

import jax
import jax.numpy as jnp

from ..core.dtype_facts import SIZE_DTYPE
from ..core.field_descriptors import FieldDescriptor
from ..core.protocol import Xtructurable
from ..core.xtructure_numpy.dataclass_ops.unique_ops import (
    _unique_mask_from_uint32_rows,
)
from .constants import SLOT_IDX_DTYPE


class BucketIdx(Xtructurable):
    index: Annotated[jax.Array, FieldDescriptor.scalar(dtype=SIZE_DTYPE)]
    slot_index: Annotated[jax.Array, FieldDescriptor.scalar(dtype=SLOT_IDX_DTYPE)]


class HashIdx(Xtructurable):
    index: Annotated[jax.Array, FieldDescriptor.scalar(dtype=SIZE_DTYPE)]


class HashTableProbe(NamedTuple):
    """Precomputed probe intermediates for a batch of states.

    These four arrays are exactly the products of the shared uint32 hash pass
    (:func:`get_new_idx_byterized_batched`): the initial bucket index (``h1``),
    the probe step (``h2``), the canonical uint32 encoding, and the fingerprint.
    Both ``lookup_parallel`` and ``parallel_insert`` derive everything they need
    from these arrays, so a caller that runs a dedup lookup immediately before an
    insert on the same batch can thread one probe through both operations instead
    of hashing the same states twice.

    Shapes for a batch of ``n`` states with ``lanes`` uint32 lanes per state:
    ``index`` ``(n,)``, ``step`` ``(n,)``, ``uint32ed`` ``(n, lanes)``,
    ``fingerprint`` ``(n,)``.
    """

    index: jax.Array
    step: jax.Array
    uint32ed: jax.Array
    fingerprint: jax.Array

    def unique_mask(
        self,
        key: jax.Array | None = None,
        filled: jax.Array | bool | None = None,
    ) -> jax.Array:
        """Return an intra-batch uniqueness mask reusing this probe's ``uint32ed`` rows."""
        batch_len = self.uint32ed.shape[0]
        if key is not None and len(key) != batch_len:
            raise ValueError(f"key length {len(key)} must match batch_len {batch_len}")
        if batch_len == 0:
            return jnp.zeros((0,), dtype=jnp.bool_)
        if filled is None:
            return _unique_mask_from_uint32_rows(self.uint32ed, key, None, batch_len)[0]
        filled_arr = jnp.asarray(filled, dtype=jnp.bool_)
        if filled_arr.ndim == 0:
            all_mask = _unique_mask_from_uint32_rows(self.uint32ed, key, None, batch_len)[0]
            return jnp.where(filled_arr, all_mask, jnp.zeros((batch_len,), dtype=jnp.bool_))
        return _unique_mask_from_uint32_rows(self.uint32ed, key, filled_arr, batch_len)[0]
