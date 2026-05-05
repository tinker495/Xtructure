"""Module for saving and loading xtructure dataclasses."""

import importlib
import os
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np

from xtructure.core.layout import get_type_layout
from xtructure.core.layout.traversal import (
    build_instance_from_leaf_values,
    iter_leaf_values,
)
from xtructure.core.layout.types import LeafLayout
from xtructure.core.protocol import Xtructurable
from xtructure.io.bitpack import from_uint8, to_uint8

METADATA_MODULE_KEY = "__xtructure_class_module__"
METADATA_CLASS_NAME_KEY = "__xtructure_class_name__"

_BITPACK_PREFIX = "__xtructure_bitpack__"


def _bitpack_keys(full_key: str) -> Tuple[str, str, str]:
    # Three separate arrays so .npz files stay inspectable and backward compatible.
    data_key = f"{full_key}.{_BITPACK_PREFIX}.data"
    shape_key = f"{full_key}.{_BITPACK_PREFIX}.shape"
    bits_key = f"{full_key}.{_BITPACK_PREFIX}.bits"
    return data_key, shape_key, bits_key


def _flatten_leaf_for_save(
    value: Any, leaf: LeafLayout, full_key: str, *, packed: bool
) -> Dict[str, np.ndarray]:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        # Layout owns the IO double-pack policy: packed field byte streams expose
        # `io_pack_bits=None`, while ordinary bit-width-constrained leaves expose
        # the bit count that should be applied to the file representation.
        io_pack_bits = leaf.io_pack_bits
        if packed and io_pack_bits is not None:
            data_key, shape_key, bits_key = _bitpack_keys(full_key)
            packed_bytes = to_uint8(jnp.asarray(value), active_bits=int(io_pack_bits))
            return {
                data_key: np.asarray(packed_bytes, dtype=np.uint8),
                shape_key: np.array(value.shape, dtype=np.int32),
                bits_key: np.array([int(io_pack_bits)], dtype=np.uint8),
            }
        return {full_key: np.asarray(value)}

    # For non-array-like fields, save as 0-dim array.
    return {full_key: np.array(value)}


def _flatten_instance_for_save(
    instance: Xtructurable, *, packed: bool = True
) -> Dict[str, np.ndarray]:
    """Flatten an xtructure instance into a dict for saving."""
    flat_data: Dict[str, np.ndarray] = {}
    for leaf, value in iter_leaf_values(instance):
        flat_data.update(_flatten_leaf_for_save(value, leaf, leaf.dotted_path, packed=packed))
    return flat_data


def save(path: str, instance: Xtructurable, *, packed: bool = True):
    """
    Saves an xtructure dataclass instance to a compressed .npz file.

    This function serializes the instance by flattening its structure and
    saving each field as a NumPy array. It also stores metadata to enable
    reconstruction of the original dataclass type upon loading.
    """
    if not hasattr(instance, "is_xtructed"):
        raise TypeError("The provided instance is not a valid xtructure dataclass.")

    data_to_save = _flatten_instance_for_save(instance, packed=packed)

    cls = instance.__class__
    data_to_save[METADATA_MODULE_KEY] = np.array(cls.__module__)
    data_to_save[METADATA_CLASS_NAME_KEY] = np.array(cls.__name__)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    np.savez_compressed(path, **data_to_save)


def _load_leaf_value(leaf: LeafLayout, data: Dict[str, Any], owner_name: str) -> Any:
    full_key = leaf.dotted_path
    data_key, shape_key, bits_key = _bitpack_keys(full_key)
    if data_key in data and shape_key in data and bits_key in data:
        bits = int(np.asarray(data[bits_key]).reshape(-1)[0])
        shape = tuple(int(x) for x in np.asarray(data[shape_key]).reshape(-1))
        packed_bytes = jnp.array(data[data_key], dtype=jnp.uint8)
        unpacked = from_uint8(packed_bytes, target_shape=shape, active_bits=bits)
        try:
            unpacked = unpacked.astype(leaf.declared_dtype)
        except TypeError:
            pass
        return unpacked

    if full_key in data:
        return jnp.array(data[full_key])

    raise KeyError(f"Missing field '{full_key}' while loading {owner_name}.")


def _unflatten_data_for_load(cls: type, data: Dict[str, Any]) -> Xtructurable:
    """Reconstruct an xtructure instance from flattened data."""
    type_layout = get_type_layout(cls)
    leaf_values = {
        leaf.path: _load_leaf_value(leaf, data, cls.__name__) for leaf in type_layout.leaves
    }
    return build_instance_from_leaf_values(cls, leaf_values)


def load(path: str) -> Xtructurable:
    """
    Loads an xtructure dataclass instance from a .npz file.

    This function reads the .npz file, reconstructs the dataclass type from
    metadata, and populates a new instance with the saved data.
    """
    with np.load(path, allow_pickle=False) as data:
        if METADATA_MODULE_KEY not in data or METADATA_CLASS_NAME_KEY not in data:
            raise ValueError("File is missing necessary xtructure metadata for loading.")

        module_name = str(data[METADATA_MODULE_KEY])
        class_name = str(data[METADATA_CLASS_NAME_KEY])

        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not find class '{class_name}' in module '{module_name}'. "
                f"Ensure the class definition is available. Original error: {e}"
            )

        return _unflatten_data_for_load(cls, data)
