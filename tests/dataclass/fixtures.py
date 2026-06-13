import jax.numpy as jnp

from tests.testdata import NestedData, SimpleData, VectorData
from xtructure import FieldDescriptor, xtructure_dataclass

__all__ = ["SimpleData", "VectorData", "NestedData", "MatrixData"]


@xtructure_dataclass
class MatrixData:
    matrix: FieldDescriptor.tensor(dtype=jnp.float32, shape=(2, 2))
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(4,), fill_value=False)
