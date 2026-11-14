"""Tests for saving and loading xtructure dataclasses."""

import os

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.testdata import NestedData, SimpleData, VectorData
from xtructure import FieldDescriptor, Xtructurable
from xtructure.core.layout.types import LeafLayout
from xtructure.io import io as io_module

# --- Test Fixtures ---


@pytest.fixture
def temp_file(tmp_path):
    """Provides a temporary file path for saving and loading."""
    file_path = tmp_path / "test_data.npz"
    yield str(file_path)
    # Cleanup is handled by pytest's tmp_path fixture


# --- Test Cases ---


@pytest.mark.parametrize(
    "cls,shape,seed",
    [
        (SimpleData, (), 0),
        (VectorData, (10,), 1),
        (NestedData, (), 2),
        (NestedData, (5, 2), 3),
    ],
)
def test_save_and_load_roundtrip(temp_file, cls, shape, seed):
    """Generic save/load roundtrip over multiple dataclass types and batch shapes."""
    key = jax.random.PRNGKey(seed)
    original_instance = cls.random(shape=shape, key=key)

    original_instance.save(temp_file)
    loaded_instance = cls.load(temp_file)

    assert isinstance(loaded_instance, cls)
    chex.assert_trees_all_equal(original_instance, loaded_instance)


def test_load_nonexistent_file():
    """Tests that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        SimpleData.load("non_existent_file_12345.npz")


def test_load_invalid_file(tmp_path):
    """Tests that loading a file without proper metadata raises ValueError."""
    invalid_file = tmp_path / "invalid.npz"
    # Save a numpy file without the required metadata keys
    import numpy as np

    np.savez_compressed(invalid_file, data=np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="File is missing necessary xtructure metadata"):
        SimpleData.load(str(invalid_file))


def test_save_to_new_directory(tmp_path):
    """Tests that save function can create a new directory."""
    dir_path = tmp_path / "new_dir"
    file_path = dir_path / "test.npz"

    assert not os.path.exists(dir_path)

    instance = SimpleData.default()
    instance.save(str(file_path))

    assert os.path.exists(file_path)


def test_load_leaf_value_cast_failure_includes_leaf_context(monkeypatch):
    """Packed leaf restore must fail loudly when declared dtype casting fails."""

    import jax.numpy as jnp
    import numpy as np

    leaf = LeafLayout(
        path=("payload",),
        declared_dtype=jnp.uint8,
        intrinsic_shape=(),
        local_intrinsic_shape=(),
        parent_intrinsic_shape=(),
        bits=8,
        packed_bits=None,
        unpacked_dtype=None,
        unpacked_intrinsic_shape=None,
        io_pack_bits=8,
    )

    monkeypatch.setattr(io_module, "from_uint8", lambda *_args, **_kwargs: object())

    data = {
        "payload.__xtructure_bitpack__.data": np.array([1], dtype=np.uint8),
        "payload.__xtructure_bitpack__.shape": np.array([], dtype=np.int32),
        "payload.__xtructure_bitpack__.bits": np.array([8], dtype=np.uint8),
    }

    with pytest.raises(TypeError, match="payload.*loading ExampleOwner"):
        io_module._load_leaf_value(leaf, data, "ExampleOwner")


@pytest.mark.parametrize(
    "position,error",
    [(np.ones((2,), dtype=np.float32), ValueError), (np.ones((3,), dtype=np.int32), TypeError)],
)
def test_raw_npz_restore_validates_stored_shape_and_dtype(tmp_path, position, error):
    # Author: tinker495. Raw NPZ preserves stored dtypes; malformed data must be
    # rejected at load, before an instance with a broken schema can escape.
    path = tmp_path / "invalid.npz"
    VectorData.default().save(str(path), packed=False)
    with np.load(path) as saved:
        payload = dict(saved)
    payload["position"] = position
    np.savez_compressed(path, **payload)
    with pytest.raises(error):
        VectorData.load(str(path))


def _require_nonzero_codes(value):
    if jnp.any(value == 0):
        raise ValueError("zero code")


class StoredCodes(Xtructurable, bitpack="off"):
    codes: FieldDescriptor.tensor(
        dtype=jnp.uint16, shape=(4,), bits=3, fill_value=1, validator=_require_nonzero_codes
    )


def test_packed_npz_converts_declared_dtype_then_checks_original_validator(tmp_path):
    # Author: tinker495. Packed IO promises an explicit declared-dtype restore;
    # that conversion remains valid, but decoded values still need validation.
    path = tmp_path / "codes.npz"
    StoredCodes.default().save(str(path))
    assert StoredCodes.load(str(path)).codes.dtype == jnp.uint16
    with np.load(path) as saved:
        payload = dict(saved)
    payload["codes.__xtructure_bitpack__.data"] = np.zeros_like(
        payload["codes.__xtructure_bitpack__.data"]
    )
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="zero code"):
        StoredCodes.load(str(path))
