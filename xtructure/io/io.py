"""Module for saving and loading xtructure dataclasses."""

import dataclasses
import importlib
import os
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

from xtructure.core.protocol import Xtructurable
from xtructure.core.xtructure_decorators.default import is_xtructure_class

METADATA_MODULE_KEY = "__xtructure_class_module__"
METADATA_CLASS_NAME_KEY = "__xtructure_class_name__"


def _flatten_instance_for_save(instance: Xtructurable, prefix: str = "") -> Dict[str, np.ndarray]:
    """Recursively flattens an xtructure instance into a dict for saving."""
    flat_data = {}
    for field in dataclasses.fields(instance):
        field_name = field.name
        value = getattr(instance, field_name)
        full_key = f"{prefix}{field_name}"

        if is_xtructure_class(type(value)):
            flat_data.update(_flatten_instance_for_save(value, prefix=f"{full_key}."))
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            flat_data[full_key] = np.asarray(value)
        else:
            # For non-array-like fields, save as 0-dim array
            flat_data[full_key] = np.array(value)
    return flat_data


def save(path: str, instance: Xtructurable):
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
    data_to_save = _flatten_instance_for_save(instance)

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
    for field in dataclasses.fields(cls):
        field_name = field.name
        full_key = f"{prefix}{field_name}"

        # The type annotation is the FieldDescriptor itself.
        # We need to inspect its `dtype` attribute for the actual type.
        field_descriptor = cls.__annotations__.get(field_name)
        if field_descriptor and is_xtructure_class(field_descriptor.dtype):
            nested_class_type = field_descriptor.dtype
            # Recursively load nested xtructure dataclass
            nested_instance_data = _unflatten_data_for_load(
                nested_class_type, data, prefix=f"{full_key}."
            )
            field_values[field_name] = nested_class_type(**nested_instance_data)
        elif full_key in data:
            # Load JAX array from NumPy array
            field_values[field_name] = jnp.array(data[full_key])
        else:
            # This case can be hit for nested structures where the keys are prefixed.
            # The recursive call handles these, so we can pass here.
            pass
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
