from collections import namedtuple
from typing import Type, TypeVar

from xtructure.core.field_descriptors import FieldDescriptor
from xtructure.core.protocol import StructuredType

T = TypeVar("T")


def add_shape_dtype_len(cls: Type[T]) -> Type[T]:
    """
    Augments the class with `shape` and `dtype` properties to inspect its
    fields, and a `__len__` method.

    The `shape` and `dtype` properties return namedtuples reflecting the
    structure of the dataclass fields.
    The `__len__` method conventionally returns the size of the first
    dimension of the first field of the instance, which is often useful
    for determining batch sizes.
    """
    shape_tuple = namedtuple("shape", ["batch"] + list(cls.__annotations__.keys()))
    field_descriptors: dict[str, FieldDescriptor] = cls.__annotations__
    default_shape = namedtuple("default_shape", cls.__annotations__.keys())(
        *[fd.intrinsic_shape for fd in field_descriptors.values()]
    )
    default_dtype = namedtuple("default_dtype", cls.__annotations__.keys())(
        *[fd.dtype for fd in field_descriptors.values()]
    )

    cls.default_shape = default_shape
    cls.default_dtype = default_dtype

    def get_shape(self) -> shape_tuple:
        """
        Returns a namedtuple containing the batch shape (if present) and the shapes of all fields.
        If a field is itself a xtructure_dataclass, its shape is included as a nested namedtuple.
        """
        # shape_tuple, cls, and default_shape are available from the outer scope (closure)
        field_shapes = []
        batch_shapes = []

        for field_name in cls.__annotations__.keys():
            field_instance = getattr(self, field_name)
            # actual_field_shape can be:
            # 1. For primitive types (e.g. np.ndarray): a simple tuple like (10, 3, 4)
            # 2. For nested xtructure_dataclass: a 'shape' namedtuple like ( (10,), (3,), (4,) )
            actual_field_shape = field_instance.shape 
            
            # default_intrinsic_shape_for_field can be:
            # 1. For primitive types: its intrinsic shape tuple, e.g. (3, 4)
            # 2. For nested xtructure_dataclass: a tuple of the intrinsic shapes of its *own* fields
            default_intrinsic_shape_for_field = getattr(default_shape, field_name)

            # Determine if the current field is itself a xtructure_dataclass
            is_nested_xtructure = (
                isinstance(actual_field_shape, tuple)
                and hasattr(actual_field_shape, "_fields") # True for namedtuples
                and actual_field_shape.__class__.__name__ == "shape" # Check if it's our specific shape namedtuple
            )

            if is_nested_xtructure:
                # actual_field_shape is the full shape of the nested instance, e.g., field_instance.shape
                # which might be: ShapeClassForNested(batch=(B_nested,), fieldA=valA, ...)

                # The batch of the nested instance contributes to the parent's batch calculation.
                batch_shapes.append(actual_field_shape.batch)

                # For the shape representation within the parent, we want to show this nested field
                # as having batch=(), but retaining its other structural field values.
                # actual_field_shape is a namedtuple. actual_field_shape[0] is its 'batch' value.
                # actual_field_shape[1:] is a tuple of its other field values in order.
                nested_field_values = actual_field_shape[1:] 
                
                # Construct the new shape object for storage in the parent:
                # ShapeClassForNested(batch=(), fieldA=valA, ...)
                processed_shape_for_field = actual_field_shape.__class__((), *nested_field_values)
                
                field_shapes.append(processed_shape_for_field)
            else:
                # Field is a primitive type (e.g., numpy array)
                # actual_field_shape is a simple tuple (e.g., (D1, D2, D3))
                # default_intrinsic_shape_for_field is a simple tuple (e.g., (S1, S2))
                
                current_processed_shape_for_field = actual_field_shape # Default if inconsistent

                if default_intrinsic_shape_for_field == ():
                    # If default intrinsic shape is empty, the entire actual shape is considered batch.
                    batch_shapes.append(actual_field_shape)
                    current_processed_shape_for_field = ()  # No specific intrinsic part
                elif len(actual_field_shape) >= len(default_intrinsic_shape_for_field) and \
                     actual_field_shape[-len(default_intrinsic_shape_for_field):] == default_intrinsic_shape_for_field:
                    # Actual shape ends with the default intrinsic shape, split it.
                    batch_part = actual_field_shape[:-len(default_intrinsic_shape_for_field)]
                    intrinsic_part = actual_field_shape[-len(default_intrinsic_shape_for_field):]
                    
                    batch_shapes.append(batch_part)
                    current_processed_shape_for_field = intrinsic_part
                else:
                    # Shapes are inconsistent or don't match.
                    batch_shapes.append(-1) # Mark batch as inconsistent
                    # current_processed_shape_for_field remains actual_field_shape by default
                
                field_shapes.append(current_processed_shape_for_field)

        # Determine final_batch_shape
        if not batch_shapes: # Handles case of no fields or if cls.__annotations__ is empty
            final_batch_shape = () 
        else:
            # Check if any field dictated an inconsistent batch (-1)
            if -1 in batch_shapes:
                final_batch_shape = -1
            else:
                # All batch_shapes are actual tuples (or empty tuples if a field's batch part is empty).
                # Check for uniformity among these potentially valid batch shapes.
                first_b_shape = batch_shapes[0]
                is_uniform = True
                for b_shape in batch_shapes[1:]:
                    if b_shape != first_b_shape:
                        is_uniform = False
                        break
                if is_uniform:
                    final_batch_shape = first_b_shape
                else:
                    final_batch_shape = -1 # Non-uniform valid batch shapes trigger inconsistency

        return shape_tuple(final_batch_shape, *field_shapes)

    setattr(cls, "shape", property(get_shape))

    type_tuple = namedtuple("dtype", cls.__annotations__.keys())

    def get_type(self) -> type_tuple:
        """Get dtypes of all fields in the dataclass"""
        return type_tuple(
            *[getattr(self, field_name).dtype for field_name in cls.__annotations__.keys()]
        )

    setattr(cls, "dtype", property(get_type))

    def get_len(self):
        """Get length of the first field's first dimension"""
        return self.shape[0][0]

    setattr(cls, "__len__", get_len)

    def get_structured_type(self) -> StructuredType:
        shape = self.shape
        if shape.batch == ():
            return StructuredType.SINGLE
        elif shape.batch == -1:
            return StructuredType.UNSTRUCTURED
        else:
            return StructuredType.BATCHED

    setattr(cls, "structured_type", property(get_structured_type))

    return cls
