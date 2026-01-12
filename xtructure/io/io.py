"""Module for saving and loading xtructure dataclasses."""

import dataclasses
import importlib
import os
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np

from xtructure.core.field_descriptors import get_field_descriptors
from xtructure.core.protocol import Xtructurable
from xtructure.core.type_utils import is_xtructure_dataclass_type
from xtructure.core.xtructure_decorators.aggregate_bitpack.bitpack import (
    from_uint8,
    to_uint8,
)

METADATA_MODULE_KEY = "__xtructure_class_module__"
METADATA_CLASS_NAME_KEY = "__xtructure_class_name__"

_BITPACK_PREFIX = "__xtructure_bitpack__"


def _bitpack_keys(full_key: str) -> Tuple[str, str, str, str]:
    # Use separate arrays so we can keep npz backward compatible and inspectable.
    data_key = f"{full_key}.{_BITPACK_PREFIX}.data"
    shape_key = f"{full_key}.{_BITPACK_PREFIX}.shape"
    bits_key = f"{full_key}.{_BITPACK_PREFIX}.bits"
    dtype_key = f"{full_key}.{_BITPACK_PREFIX}.dtype"
    return data_key, shape_key, bits_key, dtype_key


def _flatten_instance_for_save(
    instance: Xtructurable, prefix: str = "", *, packed: bool = True
) -> Dict[str, np.ndarray]:
    """Recursively flattens an xtructure instance into a dict for saving."""
    flat_data = {}
    descriptors = get_field_descriptors(instance.__class__)
    for field in dataclasses.fields(instance):
        field_name = field.name
        value = getattr(instance, field_name)
        full_key = f"{prefix}{field_name}"
        descriptor = descriptors.get(field_name)

        if is_xtructure_dataclass_type(type(value)):
            flat_data.update(
                _flatten_instance_for_save(value, prefix=f"{full_key}.", packed=packed)
            )
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            bits = getattr(descriptor, "bits", None) if descriptor is not None else None
            # If the field is already stored as packed bytes in-memory (descriptor.packed_bits),
            # do NOT apply additional IO bitpacking on top of it.
            if (
                packed
                and bits is not None
                and not (
                    descriptor is not None and getattr(descriptor, "packed_bits", None) is not None
                )
            ):
                data_key, shape_key, bits_key, dtype_key = _bitpack_keys(full_key)
                packed_bytes = to_uint8(jnp.asarray(value), active_bits=int(bits))
                flat_data[data_key] = np.asarray(packed_bytes, dtype=np.uint8)
                flat_data[shape_key] = np.asarray(np.array(value.shape, dtype=np.int32))
                flat_data[bits_key] = np.asarray(np.array([int(bits)], dtype=np.uint8))
                flat_data[dtype_key] = np.asarray(
                    np.array(str(jnp.asarray(value).dtype), dtype=np.str_)
                )
            else:
                flat_data[full_key] = np.asarray(value)
        else:
            # For non-array-like fields, save as 0-dim array
            flat_data[full_key] = np.array(value)
    return flat_data


def save(path: str, instance: Xtructurable, *, packed: bool = True):
    """
    Saves an xtructure dataclass instance to a compressed .npz file.

    This function serializes the instance by flattening its structure and
    saving each field as a NumPy array. It also stores metadata to enable
    reconstruction of the original dataclass type upon loading.

    Args:
        path: The file path (e.g., '/path/to/my_instance.npz').
        instance: The xtructure dataclass instance to save.
    """
    if not hasattr(instance, "is_xtructed"):
        raise TypeError("The provided instance is not a valid xtructure dataclass.")

    # Flatten the instance data
    data_to_save = _flatten_instance_for_save(instance, packed=packed)

    # Add metadata for reconstruction
    cls = instance.__class__
    data_to_save[METADATA_MODULE_KEY] = np.array(cls.__module__)
    data_to_save[METADATA_CLASS_NAME_KEY] = np.array(cls.__name__)

    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Save to a compressed .npz file
    np.savez_compressed(path, **data_to_save)


def _unflatten_data_for_load(cls: type, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Recursively reconstructs field values from flattened data."""
    field_values = {}
    descriptors = get_field_descriptors(cls)
    for field in dataclasses.fields(cls):
        field_name = field.name
        full_key = f"{prefix}{field_name}"

        field_descriptor = descriptors.get(field_name)
        if field_descriptor and is_xtructure_dataclass_type(field_descriptor.dtype):
            nested_class_type = field_descriptor.dtype  # type: ignore[assignment]
            # Recursively load nested xtructure dataclass
            nested_instance_data = _unflatten_data_for_load(
                nested_class_type, data, prefix=f"{full_key}."
            )
            field_values[field_name] = nested_class_type(**nested_instance_data)
        else:
            data_key, shape_key, bits_key, dtype_key = _bitpack_keys(full_key)
            if data_key in data and shape_key in data and bits_key in data:
                bits = int(np.asarray(data[bits_key]).reshape(-1)[0])
                shape = tuple(int(x) for x in np.asarray(data[shape_key]).reshape(-1))
                packed_bytes = jnp.array(data[data_key], dtype=jnp.uint8)
                unpacked = from_uint8(packed_bytes, target_shape=shape, active_bits=bits)
                # Cast back to declared dtype when possible.
                if field_descriptor is not None and not is_xtructure_dataclass_type(
                    field_descriptor.dtype
                ):
                    try:
                        unpacked = unpacked.astype(field_descriptor.dtype)
                    except TypeError:
                        pass
                field_values[field_name] = unpacked
            elif full_key in data:
                # Load JAX array from NumPy array
                field_values[field_name] = jnp.array(data[full_key])
    return field_values


def load(path: str) -> Xtructurable:
    """
    Loads an xtructure dataclass instance from a .npz file.

    This function reads the .npz file, reconstructs the dataclass type from
    metadata, and populates a new instance with the saved data.

    Args:
        path: The file path of the .npz file to load.

    Returns:
        A new instance of the saved xtructure dataclass.
    """
    with np.load(path, allow_pickle=False) as data:
        # Extract metadata
        if METADATA_MODULE_KEY not in data or METADATA_CLASS_NAME_KEY not in data:
            raise ValueError("File is missing necessary xtructure metadata for loading.")

        module_name = str(data[METADATA_MODULE_KEY])
        class_name = str(data[METADATA_CLASS_NAME_KEY])

        # Dynamically import the class
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not find class '{class_name}' in module '{module_name}'. "
                f"Ensure the class definition is available. Original error: {e}"
            )

        # Reconstruct the instance from flattened data
        instance_data = _unflatten_data_for_load(cls, data)
        return cls(**instance_data)
