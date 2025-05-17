from enum import Enum


# enum for state type
class StructuredType(Enum):
    SINGLE = 0
    BATCHED = 1
    UNSTRUCTURED = 2
