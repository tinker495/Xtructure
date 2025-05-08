import chex
import jax
import jax.numpy as jnp

from collections import namedtuple
from typing import TypeVar, Type, Dict, Any
from enum import Enum
from tabulate import tabulate

MAX_PRINT_BATCH_SIZE = 4
SHOW_BATCH_SIZE = 2

T = TypeVar("T")


def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)

def get_leaf_elements(tree: chex.Array):
    """
    Extracts leaf elements from a nested tuple structure.

    Args:
        data: The nested tuple (or a single element).

    Yields:
        Leaf elements (non-tuple elements) within the nested structure.
    """
    if isnamedtupleinstance(tree):
        for item in tree:
            yield from get_leaf_elements(item)  # Recursively process sub-tuples
    else:
        yield tree  # Yield the leaf element

def xtructure_data(cls):
    """
    Decorator that creates a dataclass for HeapValue with additional functionality.
    Adds shape, dtype, getitem, and len properties to the class.

    Args:
        cls: The class to be decorated

    Returns:
        The decorated class with additional heap value functionality
    """
    cls = chex.dataclass(cls)

    # add shape and dtype and getitem and len
    cls = add_shape_dtype_getitem_len(cls)
    cls = add_default_parser(cls)
    cls = add_string_parser(cls)

    # Ensure class has a default method for initialization
    assert hasattr(cls, "default"), "HeapValue class must have a default method."

    return cls

# enum for state type
class StructuredType(Enum):
    SINGLE = 0
    BATCHED = 1
    UNSTRUCTURED = 2

def add_shape_dtype_getitem_len(cls: Type[T]) -> Type[T]:
    shape_tuple = namedtuple("shape", cls.__annotations__.keys())

    def get_shape(self) -> shape_tuple:
        """Get shapes of all fields in the dataclass"""
        return shape_tuple(
            *[getattr(self, field_name).shape for field_name in cls.__annotations__.keys()]
        )

    setattr(cls, "shape", property(get_shape))

    type_tuple = namedtuple("dtype", cls.__annotations__.keys())

    def get_type(self) -> type_tuple:
        """Get dtypes of all fields in the dataclass"""
        return type_tuple(
            *[
                jnp.dtype(getattr(self, field_name).dtype)
                for field_name in cls.__annotations__.keys()
            ]
        )

    setattr(cls, "dtype", property(get_type))

    def getitem(self, index):
        """Support indexing operations on the dataclass"""
        new_values = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, "__getitem__"):
                new_values[field_name] = field_value[index]
            else:
                new_values[field_name] = field_value
        return cls(**new_values)

    setattr(cls, "__getitem__", getitem)

    def len(self):
        """Get length of the first field's first dimension"""
        return self.shape[0][0]

    setattr(cls, "__len__", len)
    return cls


def add_default_parser(cls: Type[T]) -> Type[T]:
    """
    This function is a decorator that adds a default dataclass to the class.
    this function for making a default dataclass with the given shape, for example, hash table of the puzzle.
    """
    assert hasattr(cls, "default"), "There is no default method."

    default_shape = cls.default().shape
    try:
        default_dim = len(default_shape[0])
    except IndexError:
        default_dim = None
        """
        if default_dim is None, it means that the default shape is not a batch.
        """
        return cls

    def get_default_shape(self) -> Dict[str, Any]:
        return default_shape

    def get_structured_type(self) -> StructuredType:
        shape = self.shape
        if shape == default_shape:
            return StructuredType.SINGLE
        elif all(
            ds == s[-max(len(ds), 1) :] or (ds == () and len(s) == 1)
            for ds, s in zip(get_leaf_elements(default_shape), get_leaf_elements(shape))
        ):
            return StructuredType.BATCHED
        else:
            return StructuredType.UNSTRUCTURED

    def batch_shape(self) -> tuple[int, ...]:
        if self.structured_type == StructuredType.BATCHED:
            shape = list(get_leaf_elements(self.shape))
            return shape[0][:-default_dim]
        else:
            raise ValueError(f"State is not structured: {self.shape} != {self.default_shape}")

    def reshape(self, new_shape: tuple[int, ...]) -> T:
        if self.structured_type == StructuredType.BATCHED:
            total_length = jnp.prod(jnp.array(self.batch_shape))
            new_total_length = jnp.prod(jnp.array(new_shape))
            batch_dim = len(self.batch_shape)
            if total_length != new_total_length:
                raise ValueError(
                    f"Total length of the state and new shape does not match: {total_length} != {new_total_length}"
                )
            return jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, new_shape + x.shape[batch_dim:]), self
            )
        else:
            raise ValueError(f"State is not structured: {self.shape} != {self.default_shape}")

    def flatten(self):
        total_length = jnp.prod(jnp.array(self.batch_shape))
        return jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (total_length, *x.shape[-default_dim:])), self
        )

    # add method based on default state
    setattr(cls, "default_shape", property(get_default_shape))
    setattr(cls, "structured_type", property(get_structured_type))
    setattr(cls, "batch_shape", property(batch_shape))
    setattr(cls, "reshape", reshape)
    setattr(cls, "flatten", flatten)
    return cls

def add_string_parser(cls: Type[T]) -> Type[T]:
    """
    This function is a decorator that adds a __str__ method to
    the class that returns a string representation of the class.
    """

    if not hasattr(cls, "__str__"):
        parsefunc = cls.__str__
    else:
        parsefunc = lambda x: str(x)

    def get_str(self, **kwargs) -> str:
        structured_type = self.structured_type

        if structured_type == StructuredType.SINGLE:
            return parsefunc(self, **kwargs)
        elif structured_type == StructuredType.BATCHED:
            batch_shape = self.batch_shape
            batch_len = (
                jnp.prod(jnp.array(batch_shape)) if len(batch_shape) != 1 else batch_shape[0]
            )
            results = []
            if batch_len <= MAX_PRINT_BATCH_SIZE:
                for i in range(batch_len):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                    kwargs_idx = {k: v[index] for k, v in kwargs.items()}
                    results.append(parsefunc(current_state, **kwargs_idx))
            else:
                for i in range(SHOW_BATCH_SIZE):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                    kwargs_idx = {k: v[index] for k, v in kwargs.items()}
                    results.append(parsefunc(current_state, **kwargs_idx))
                results.append("...\n(batch : " + f"{batch_shape})")
                for i in range(batch_len - SHOW_BATCH_SIZE, batch_len):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state = jax.tree_util.tree_map(lambda x: x[index], self)
                    kwargs_idx = {k: v[index] for k, v in kwargs.items()}
                    results.append(parsefunc(current_state, **kwargs_idx))
            return tabulate([results], tablefmt="plain")
        else:
            raise ValueError(f"State is not structured: {self.shape} != {self.default_shape}")

    setattr(cls, "__str__", get_str)
    setattr(cls, "str", get_str)
    return cls