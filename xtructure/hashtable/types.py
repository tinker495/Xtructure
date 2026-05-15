from ..core import FieldDescriptor, xtructure_dataclass
from ..core.container_facts import SIZE_DTYPE
from .constants import SLOT_IDX_DTYPE


@xtructure_dataclass
class BucketIdx:
    index: FieldDescriptor.scalar(dtype=SIZE_DTYPE)
    slot_index: FieldDescriptor.scalar(dtype=SLOT_IDX_DTYPE)


@xtructure_dataclass
class HashIdx:
    index: FieldDescriptor.scalar(dtype=SIZE_DTYPE)
