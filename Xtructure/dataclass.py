from functools import partial
import chex
import jax
import jax.numpy as jnp
import typing # Added for Protocol and TYPE_CHECKING

from collections import namedtuple
from typing import TypeVar, Type, Dict, Any, Tuple as TypingTuple, Protocol, Callable # Added Protocol, TypingTuple, Callable
from enum import Enum
from tabulate import tabulate

# Import FieldDescriptor and inspect
from .field_descriptors import FieldDescriptor
import inspect

MAX_PRINT_BATCH_SIZE = 4
SHOW_BATCH_SIZE = 2

T = TypeVar("T")


# Protocol defining the interface added by @xtructure_data
class Xtructurable(Protocol[T]):
    # Fields from the original class that chex.dataclass would process
    # These are implicitly part of T. For the protocol to be complete,
    # it assumes T will have __annotations__.
    __annotations__: Dict[str, Any]
    # __dict__ is used by the __getitem__ implementation
    __dict__: Dict[str, Any]


    # Methods and properties added by add_shape_dtype_getitem_len
    @property
    def shape(self) -> Any: # Actual type is a dynamically generated namedtuple
        ...
    @property
    def dtype(self) -> Any: # Actual type is a dynamically generated namedtuple
        ...
    def __getitem__(self: T, index: Any) -> T:
        ...
    def __len__(self) -> int:
        ...

    # Methods and properties added by add_structure_utilities_and_random
    # Assumes the class T has a 'default' classmethod as per the decorator's assertion
    @classmethod
    def default(cls: Type[T], shape: Any = ...) -> T:
        ...
    @property
    def default_shape(self) -> Any: # Derived from self.default().shape
        ...
    @property
    def structured_type(self) -> 'StructuredType': # Forward reference for StructuredType
        ...
    @property
    def batch_shape(self) -> TypingTuple[int, ...]:
        ...
    def reshape(self: T, new_shape: TypingTuple[int, ...]) -> T:
        ...
    def flatten(self: T) -> T:
        ...
    @classmethod
    def random(cls: Type[T], shape: TypingTuple[int, ...] = ..., key: Any = ...) -> T: # Ellipsis for default value
        ...

    # Methods and properties added by add_string_representation_methods
    def __str__(self) -> str: # The actual implementation takes **kwargs, but signature can be simpler for Protocol
        ...
    def str(self) -> str: # Alias for __str__
        ...


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
    Extracts leaf elements from a nested structure, typically composed of
    namedtuples and JAX arrays at the leaves.

    Args:
        tree: The nested structure (e.g., a namedtuple containing other
              namedtuples or JAX arrays, or a single JAX array).

    Yields:
        Leaf elements (non-namedtuple elements) within the nested structure.
    """
    if isnamedtupleinstance(tree):
        for item in tree:
            yield from get_leaf_elements(item)  # Recursively process sub-tuples
    else:
        yield tree  # Yield the leaf element

def xtructure_dataclass(cls: Type[T]) -> Type[Xtructurable[T]]:
    """
    Decorator that ensures the input class is a `chex.dataclass` (or converts
    it to one) and then augments it with additional functionality related to its
    structure, type, and operations like indexing, default instance creation,
    random instance generation, and string representation.

    It adds properties like `shape`, `dtype`, `default_shape`, `structured_type`,
    `batch_shape`, and methods like `__getitem__`, `__len__`, `reshape`,
    `flatten`, `random`, and `__str__`.

    Args:
        cls: The class to be decorated. It is expected to have a `default`
             classmethod for some functionalities.

    Returns:
        The decorated class with the aforementioned additional functionalities.
    """
    cls = chex.dataclass(cls)

    # Ensure class has a default method for initialization
    cls = _add_auto_default_method_if_needed(cls)

    # add shape and dtype and getitem and len
    cls = add_shape_dtype_getitem_len(cls)
    cls = add_structure_utilities_and_random(cls) # Renamed from add_default_parser
    cls = add_string_representation_methods(cls) # Renamed from add_string_parser

    # Ensure class has a default method for initialization
    assert hasattr(cls, "default"), "HeapValue class must have a default method."

    return cls

# enum for state type
class StructuredType(Enum):
    SINGLE = 0
    BATCHED = 1
    UNSTRUCTURED = 2

def add_shape_dtype_getitem_len(cls: Type[T]) -> Type[T]:
    """
    Augments the class with properties to inspect the shape and dtype of its
    fields, an `__getitem__` method for indexing/slicing, and a `__len__`
    method.

    The `shape` and `dtype` properties return namedtuples reflecting the
    structure of the dataclass fields.
    The `__getitem__` method allows instances to be indexed, applying the
    index to each field.
    The `__len__` method conventionally returns the size of the first
    dimension of the first field of the instance, which is often useful
    for determining batch sizes.
    """
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
                getattr(self, field_name).dtype
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


def add_structure_utilities_and_random(cls: Type[T]) -> Type[T]:
    """
    Augments the class with utility methods and properties related to its
    structural representation (based on a 'default' instance), batch operations,
    and random instance generation.

    Requires the class to have a `default` classmethod, which is used to
    determine default shapes, dtypes, and behaviors.

    Adds:
        - Properties:
            - `default_shape`: Shape of the instance returned by `cls.default()`.
            - `structured_type`: An enum (`StructuredType`) indicating if the
              instance is SINGLE, BATCHED, or UNSTRUCTURED relative to its
              default shape.
            - `batch_shape`: The shape of the batch dimensions if `structured_type`
              is BATCHED.
        - Instance Methods:
            - `reshape(new_shape)`: Reshapes the batch dimensions of a BATCHED instance.
            - `flatten()`: Flattens the batch dimensions of a BATCHED instance.
        - Classmethod:
            - `random(shape=(), key=None)`: Generates an instance with random data.
              The `shape` argument specifies the desired batch shape, which is
              prepended to the default field shapes.
    """
    assert hasattr(cls, "default"), "There is no default method."

    default_shape = cls.default().shape
    default_dtype = cls.default().dtype
    try:
        # Get the shape of the first leaf element in the default instance
        first_leaf_shape = next(get_leaf_elements(default_shape))
        default_dim = len(first_leaf_shape)
    except StopIteration: # No leaf elements (e.g., class with no fields)
        default_dim = 0
    except IndexError: # Should ideally be caught by StopIteration if get_leaf_elements is robust
        # This case was for when default_shape[0] was accessed on an empty default_shape.
        # With get_leaf_elements, StopIteration is more likely for truly empty structures.
        default_dim = 0 # Defaulting to 0 for safety, implies scalar-like leaves.

    # Pre-calculate generation configurations for the random method
    _field_generation_configs = []
    # Ensure consistent order for key splitting, matching __annotations__
    _field_names_for_random = list(cls.__annotations__.keys()) 

    for field_name_cfg in _field_names_for_random:
        cfg = {}
        cfg['name'] = field_name_cfg
        # Retrieve the dtype or nested dtype tuple for the current field
        actual_dtype_or_nested_dtype_tuple = getattr(default_dtype, field_name_cfg)
        cfg['default_field_shape'] = getattr(default_shape, field_name_cfg, ()) # Default to empty tuple if not found
        
        if isnamedtupleinstance(actual_dtype_or_nested_dtype_tuple):
            # This field is a nested xtructure_data instance
            cfg['type'] = 'xtructure'
            # Store the actual nested class type (e.g., Parent, Current)
            cfg['nested_class_type'] = cls.__annotations__[field_name_cfg]
            # Store the namedtuple of dtypes for the nested structure
            cfg['actual_dtype'] = actual_dtype_or_nested_dtype_tuple 
        else:
            # This field is a regular JAX array
            actual_dtype = actual_dtype_or_nested_dtype_tuple # It's a single JAX dtype here
            cfg['actual_dtype'] = actual_dtype # Store the single JAX dtype

            if jnp.issubdtype(actual_dtype, jnp.integer):
                cfg['type'] = 'bits_int' # Unified type for all full-range integers via bits
                if jnp.issubdtype(actual_dtype, jnp.unsignedinteger):
                    cfg['bits_gen_dtype'] = actual_dtype # Generate bits of this same unsigned type
                    cfg['view_as_signed'] = False
                else: # It's a signed integer
                    unsigned_equivalent_str = f'uint{actual_dtype.itemsize * 8}'
                    cfg['bits_gen_dtype'] = jnp.dtype(unsigned_equivalent_str) # Generate bits of corresponding unsigned type
                    cfg['view_as_signed'] = True # And then view them as the actual signed type
            elif jnp.issubdtype(actual_dtype, jnp.floating):
                cfg['type'] = 'float'
                cfg['gen_dtype'] = actual_dtype 
            elif actual_dtype == jnp.bool_:
                cfg['type'] = 'bool'
            else:
                cfg['type'] = 'other' # Fallback
                cfg['gen_dtype'] = actual_dtype
        _field_generation_configs.append(cfg)


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
            return shape[0][:(len(shape[0])-default_dim)]
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
    
    def random(cls, shape=(), key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        
        data = {}
        keys = jax.random.split(key, len(_field_generation_configs))

        for i, cfg in enumerate(_field_generation_configs):
            field_key = keys[i]
            field_name = cfg['name']
            
            if cfg['type'] == 'xtructure':
                nested_class = cfg['nested_class_type']
                # Recursively call random for the nested xtructure_data class.
                # Pass the batch 'shape' and field_key.
                # The nested random method will manage its own internal field shapes.
                data[field_name] = nested_class.random(shape=shape, key=field_key)
            else:
                # This branch handles primitive JAX array fields.
                current_default_shape = cfg['default_field_shape']
                if not isinstance(current_default_shape, tuple):
                    current_default_shape = (current_default_shape,) # Ensure it's a tuple for concatenation
                
                target_shape = shape + current_default_shape

                if cfg['type'] == 'bits_int':
                    generated_bits = jax.random.bits(
                        field_key,
                        shape=target_shape,
                        dtype=cfg['bits_gen_dtype']
                    )
                    if cfg['view_as_signed']:
                        data[field_name] = generated_bits.view(cfg['actual_dtype'])
                    else:
                        data[field_name] = generated_bits
                elif cfg['type'] == 'float':
                    data[field_name] = jax.random.uniform(
                        field_key, target_shape, dtype=cfg['gen_dtype']
                    )
                elif cfg['type'] == 'bool':
                    data[field_name] = jax.random.bernoulli(
                        field_key, shape=target_shape # p=0.5 by default
                    )
                else: # Fallback for 'other' dtypes (cfg['type'] == 'other')
                    try:
                        data[field_name] = jnp.zeros(target_shape, dtype=cfg['gen_dtype'])
                    except TypeError:
                        raise NotImplementedError(
                            f"Random generation for dtype {cfg['gen_dtype']} (field: {field_name}) is not implemented robustly."
                        )
        return cls(**data)

    # add method based on default state
    setattr(cls, "default_shape", property(get_default_shape))
    setattr(cls, "structured_type", property(get_structured_type))
    setattr(cls, "batch_shape", property(batch_shape))
    setattr(cls, "reshape", reshape)
    setattr(cls, "flatten", flatten)
    setattr(cls, "random", classmethod(random))
    return cls

def add_string_representation_methods(cls: Type[T]) -> Type[T]:
    """
    Adds custom `__str__` and `str` methods to the class for generating
    a more informative string representation.

    It handles instances categorized by `structured_type` differently:
    - `SINGLE`: Uses the original `__str__` (or `repr` if basic) of the instance.
    - `BATCHED`: Provides a summarized view if the batch is large, showing
      the first few and last few elements, along with the batch shape.
      Uses `tabulate` for formatting.
    - `UNSTRUCTURED`: Indicates that the data is unstructured relative to its
      default shape.
    """

    # Capture the class's __str__ method as it exists *before* this decorator replaces it.
    # This will typically be the __str__ provided by chex.dataclass (similar to its __repr__),
    # or a user-defined __str__ if the user added one before @xtructure_data.
    _original_str_method = getattr(cls, '__str__', None)

    # Determine the function to use for formatting a single item.
    # If the original __str__ is just the basic one from `object`, it's not very informative.
    # In such cases, or if no __str__ was found, `repr` is a better fallback for dataclasses.
    if _original_str_method is None or _original_str_method == object.__str__:
        _single_item_formatter = repr
    else:
        # Use the captured original __str__ method.
        # We need to ensure it's called as a method of the item.
        _single_item_formatter = lambda item, **k: _original_str_method(item, **k)
        # Note: Original __str__ methods typically don't take **kwargs.
        # If kwargs support is needed for the single item formatter,
        # the user would need to define a specific method and the decorator would look for that.
        # For now, we assume the original __str__ doesn't use kwargs from get_str.

    def get_str(self, use_kwargs: bool = False, **kwargs) -> str:
        # This 'self' is an instance of the decorated class 'cls'
        # 'kwargs' are passed from the print(instance) or str(instance) call.

        structured_type = self.structured_type # This must be a valid property

        if structured_type == StructuredType.SINGLE:
            # For a single item, call the chosen formatter.
            # Pass kwargs only if the formatter is not the built-in repr.
            if _single_item_formatter is repr:
                if use_kwargs:
                    return repr(self, **kwargs)
                else:
                    return repr(self)
            else:
                # Our lambda wrapper for _original_str_method doesn't currently pass kwargs.
                # If _original_str_method was, e.g., a user's custom __str__ that took kwargs,
                # this would need adjustment or a different convention.
                # For now, assuming _original_str_method (like a dataclass __str__) doesn't expect these kwargs.
                if use_kwargs:
                    return _single_item_formatter(self, **kwargs)
                else:
                    return _single_item_formatter(self) # Invokes the lambda: _original_str_method(self)
        
        elif structured_type == StructuredType.BATCHED:
            batch_shape = self.batch_shape
            batch_len_val = (
                jnp.prod(jnp.array(batch_shape)) if len(batch_shape) != 1 else batch_shape[0]
            )
            py_batch_len = int(batch_len_val)

            results = []
            if py_batch_len <= MAX_PRINT_BATCH_SIZE:
                for i in range(py_batch_len):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state_slice = self[index]
                    # kwargs_idx = {k: v[index] for k, v in kwargs.items()} # Index kwargs if they are batched
                    # For now, assume single_item_formatter doesn't use these indexed kwargs
                    if use_kwargs:
                        results.append(_single_item_formatter(current_state_slice, **kwargs))
                    else:
                        results.append(_single_item_formatter(current_state_slice))
            else:
                for i in range(SHOW_BATCH_SIZE):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state_slice = self[index]
                    if use_kwargs:
                        results.append(_single_item_formatter(current_state_slice, **kwargs))
                    else:
                        results.append(_single_item_formatter(current_state_slice))
                
                results.append("...\n(batch : " + f"{batch_shape})")
                
                for i in range(py_batch_len - SHOW_BATCH_SIZE, py_batch_len):
                    index = jnp.unravel_index(i, batch_shape)
                    current_state_slice = self[index]
                    if use_kwargs:
                        results.append(_single_item_formatter(current_state_slice, **kwargs))
                    else:
                        results.append(_single_item_formatter(current_state_slice))
            return tabulate([results], tablefmt="plain")
        else: # UNSTRUCTURED or any other case
            # Fallback for unstructured or unexpected types to avoid errors,
            # or re-raise the original error if preferred.
            # The original code raised: ValueError(f"State is not structured: {self.shape} != {self.default_shape}")
            # Using repr as a safe fallback:
            return f"<Unstructured {cls.__name__} data, shape: {self.shape}, default_shape: {self.default_shape}>"


    setattr(cls, "__str__", lambda self, **kwargs: get_str(self, use_kwargs=False, **kwargs))
    setattr(cls, "str", lambda self, **kwargs: get_str(self, use_kwargs=True, **kwargs)) # Alias .str to the new __str__
    return cls

# Helper function to check if an annotation object is a class (potential nested xtructure)
# For the purpose of _create_default_method, we check hasattr(default) at runtime.
def _is_potentially_xtructure_class(annotation_obj: Any) -> bool:
    return inspect.isclass(annotation_obj)

def _create_default_method(cls_to_modify: Type[T]) -> Callable[..., T]:
    @classmethod
    def auto_default(cls: Type[T], shape: TypingTuple[int, ...] = ()) -> T:
        default_values: Dict[str, Any] = {}
        annotations = getattr(cls, '__annotations__', {})

        for field_name, annotation_obj in annotations.items():
            if isinstance(annotation_obj, FieldDescriptor):
                descriptor = annotation_obj
                dtype_of_field_descriptor = descriptor.dtype

                if _is_potentially_xtructure_class(dtype_of_field_descriptor):
                    # dtype_of_field_descriptor is a CLASS (e.g. <class 'numpy.int32'> or <class '__main__.Parent'>)
                    # Differentiate: is it a JAX primitive *class* or a user-defined xtructure *class*?
                    is_jax_primitive_type_class = False
                    try:
                        if jnp.issubdtype(dtype_of_field_descriptor, jnp.number) or \
                           jnp.issubdtype(dtype_of_field_descriptor, jnp.bool_):
                            is_jax_primitive_type_class = True
                    except TypeError: # Not a type that jnp.issubdtype recognizes as a primitive base
                        is_jax_primitive_type_class = False
                    
                    if is_jax_primitive_type_class:
                        # It's like jnp.int32, jnp.float32. Use jnp.full.
                        intrinsic_shape = descriptor.intrinsic_shape if isinstance(descriptor.intrinsic_shape, tuple) else (descriptor.intrinsic_shape,)
                        field_shape = shape + intrinsic_shape
                        default_values[field_name] = jnp.full(
                            field_shape,
                            descriptor.fill_value,
                            dtype=dtype_of_field_descriptor # Use the primitive class directly
                        )
                    else:
                        # It's a user-defined class like Parent. Use its .default() method.
                        nested_class_type = dtype_of_field_descriptor
                        if not hasattr(nested_class_type, 'default'):
                            raise TypeError(
                                f"Runtime error in auto-generated .default() for '{cls.__name__}': "
                                f"Nested field '{field_name}' (type '{nested_class_type.__name__}' via FieldDescriptor.dtype) "
                                f"does not have a .default() method. Ensure it's an @xtructure_data class."
                            )
                        default_values[field_name] = nested_class_type.default(shape=shape)
                elif isinstance(dtype_of_field_descriptor, jnp.dtype):
                    # dtype_of_field_descriptor is a JAX dtype INSTANCE (e.g., jnp.dtype('int32')). Use jnp.full.
                    intrinsic_shape = descriptor.intrinsic_shape if isinstance(descriptor.intrinsic_shape, tuple) else (descriptor.intrinsic_shape,)
                    field_shape = shape + intrinsic_shape
                    default_values[field_name] = jnp.full(
                        field_shape,
                        descriptor.fill_value,
                        dtype=dtype_of_field_descriptor
                    )
                else:
                    # FieldDescriptor.dtype is neither a recognized class nor a jnp.dtype instance.
                    raise TypeError(
                        f"Runtime error in auto-generated .default() for '{cls.__name__}': "
                        f"Field '{field_name}' uses FieldDescriptor with an unsupported .dtype attribute: '{dtype_of_field_descriptor}' "
                        f"(type: {type(dtype_of_field_descriptor).__name__}). Expected a JAX primitive type/class "
                        f"(like jnp.int32 or jnp.dtype('int32')), or an @xtructure_data class type (like Parent)."
                    )
            elif _is_potentially_xtructure_class(annotation_obj):
                # annotation_obj is the type of the nested class directly (e.g., parent: Parent)
                nested_class_type = annotation_obj
                if not hasattr(nested_class_type, 'default'):
                    raise TypeError(
                        f"Runtime error in auto-generated .default() for '{cls.__name__}': "
                        f"Nested field '{field_name}' of type '{nested_class_type.__name__}' "
                        f"does not have a .default() method. Ensure it's an @xtructure_data class."
                    )
                default_values[field_name] = nested_class_type.default(shape=shape)
            else:
                # This case should ideally be caught by the pre-flight check in _add_auto_default_method_if_needed
                # However, this runtime check is a safeguard.
                raise TypeError(
                    f"Runtime error in auto-generated .default() for '{cls.__name__}': "
                    f"Field '{field_name}' with annotation '{annotation_obj}' (type: {type(annotation_obj).__name__}) "
                    f"is not a FieldDescriptor or a compatible nested xtructure_data class."
                )
        
        # Handle fields that are part of the dataclass but not in annotations (e.g. field_name: type = default_value)
        # These fields should pick up their class-defined defaults automatically when cls(**default_values) is called,
        # as long as they are not required in the __init__ generated by dataclass.
        # If they are required (no default value provided in class def), and not in annotations,
        # cls(**default_values) will fail, which is correct behavior.
        return cls(**default_values)
    return auto_default

def _add_auto_default_method_if_needed(cls: Type[T]) -> Type[T]:
    if hasattr(cls, "default"):
        # User has provided a custom default method, so we don't overwrite it.
        return cls

    annotations = getattr(cls, '__annotations__', {})
    
    # Case 1: No annotations and no actual fields (e.g. `class A: pass`)
    # __dataclass_fields__ is populated by chex.dataclass() which is called *before* this function.
    dataclass_fields = getattr(cls, '__dataclass_fields__', {})
    if not annotations and not dataclass_fields:
        # Safe to generate a simple default() that calls cls()
        setattr(cls, "default", _create_default_method(cls))
        return cls

    # Case 2: Has annotations. Check if all are suitable.
    if annotations: # Only proceed if there are annotations to check
        for field_name, ann_obj in annotations.items():
            is_fd = isinstance(ann_obj, FieldDescriptor)
            is_potential_nested = _is_potentially_xtructure_class(ann_obj)

            if not is_fd and not is_potential_nested:
                # Found an annotation that is not a FieldDescriptor and not a class.
                # Auto-generation cannot handle this. Do not attach auto_default.
                # The assertion in xtructure_data will provide a detailed error.
                return cls
        
        # All annotations are suitable (FieldDescriptor or a class type).
        setattr(cls, "default", _create_default_method(cls))
        return cls
    
    # Case 3: No annotations, but has dataclass fields (e.g. `my_field = 1` or `my_field: int` if not in __annotations__ somehow)
    # In this scenario, we can't auto-generate based on FieldDescriptors.
    # The class must provide its own default method.
    # So, we don't attach anything, and the main assertion in xtructure_data will trigger.
    return cls