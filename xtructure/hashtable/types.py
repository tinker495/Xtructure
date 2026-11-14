from typing import Annotated, NamedTuple

import jax

from ..core.dtype_facts import SIZE_DTYPE
from ..core.field_descriptors import FieldDescriptor
from ..core.protocol import Xtructurable
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
