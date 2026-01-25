from typing import Annotated

import chex

from ..core import FieldDescriptor, xtructure_dataclass
from .constants import SIZE_DTYPE, SLOT_IDX_DTYPE

_BUCKET_INDEX_DESC = FieldDescriptor.scalar(dtype=SIZE_DTYPE)
_BUCKET_SLOT_INDEX_DESC = FieldDescriptor.scalar(dtype=SLOT_IDX_DTYPE)
_HASH_INDEX_DESC = FieldDescriptor.scalar(dtype=SIZE_DTYPE)


@xtructure_dataclass
class BucketIdx:
    index: Annotated[chex.Array, _BUCKET_INDEX_DESC]
    slot_index: Annotated[chex.Array, _BUCKET_SLOT_INDEX_DESC]


@xtructure_dataclass
class HashIdx:
    index: Annotated[chex.Array, _HASH_INDEX_DESC]


__all__ = ["BucketIdx", "HashIdx"]
