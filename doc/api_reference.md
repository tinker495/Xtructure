# API Reference: Other Exposed Types and Constants

Beyond the main data structures (`HashTable`, `BGPQ`) and the dataclass utilities (`@xtructure_dataclass`, `FieldDescriptor`), the Xtructure library also exposes several other types and constants primarily from `Xtructure.annotate` and `Xtructure.dataclass` via its `Xtructure/__init__.py` file. These are often used internally by the main components but can be useful for type hinting or understanding the underlying mechanics.

## From `Xtructure.annotate`

These are mostly JAX dtype definitions and constants influencing the behavior and sizing of the data structures.

*   **`KEY_DTYPE`**: The default JAX dtype for keys. Crucial for `BGPQ` where keys are typically floating-point numbers (e.g., `jnp.float32`, `jnp.float64`) due to the use of `jnp.inf` for padding.
*   **`SIZE_DTYPE`**: The JAX dtype used for representing sizes and capacities within the data structures (e.g., `jnp.int32`).
*   **`HASH_POINT_DTYPE`**: Dtype related to hash values in `HashTable`.
*   **`HASH_TABLE_IDX_DTYPE`**: Dtype for indices within the `HashTable`'s internal Cuckoo table structure.
*   **`ACTION_DTYPE`**: (Purpose might be more specific to internal logic, e.g., for reinforcement learning or stateful operations if applicable beyond current structures).

*   **`CUCKOO_TABLE_N`**: An integer constant defining the number of hash functions (and thus possible locations) used by the Cuckoo hashing algorithm in `HashTable`. A typical small integer (e.g., 2, 3, or 4).
*   **`HASH_SIZE_MULTIPLIER`**: A floating-point constant used in `HashTable.build()` to determine the internal table size relative to the user-requested `capacity`. The internal capacity is calculated roughly as `(HASH_SIZE_MULTIPLIER * capacity) / CUCKOO_TABLE_N` to provide enough space for Cuckoo hashing to work effectively.

## From `Xtructure.dataclass`

These relate to the dataclass system provided by Xtructure.

*   **`Xtructurable`**: This is a `typing.Protocol` that classes decorated with `@xtructure_dataclass` will conform to. It defines the set of methods and properties that `@xtructure_dataclass` adds (e.g., `.shape`, `.dtype`, `.default()`, `.random()`, `__getitem__`, etc.). Useful for type hinting if you want to indicate that a function expects an Xtructure dataclass instance without specifying a concrete class.

*   **`StructuredType`**: An `enum.Enum` with members:
    *   `StructuredType.SINGLE`: Indicates an instance has a scalar structure (matching its default non-batched form).
    *   `StructuredType.BATCHED`: Indicates an instance has one or more batch dimensions prepended to its fields' intrinsic shapes.
    *   `StructuredType.UNSTRUCTURED`: Indicates an instance's shape doesn't conform to either `SINGLE` or `BATCHED` relative to its definition (e.g., if fields have inconsistent batching).
    This enum is the return type of the `structured_type` property of an `@xtructure_dataclass` instance. 