from ..core.dtype_facts import SIZE_DTYPE
from ..core.field_descriptors import FieldDescriptor
from ..core.xtructure_decorators import xtructure_dataclass
from .constants import SLOT_IDX_DTYPE


@xtructure_dataclass
class BucketIdx:
    index: FieldDescriptor.scalar(dtype=SIZE_DTYPE)
    slot_index: FieldDescriptor.scalar(dtype=SLOT_IDX_DTYPE)


@xtructure_dataclass
class HashIdx:
    index: FieldDescriptor.scalar(dtype=SIZE_DTYPE)
