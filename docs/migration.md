# Xtructurable Migration

`xtructure_dataclass` has been removed from `xtructure`, `xtructure.core`, and its
implementation module. Existing imports fail immediately. Define structures by
inheriting directly from `Xtructurable`.

## Replace the decorator with inheritance

Remove the decorator and its import, add `Xtructurable` as a base, and move the
same `validate` and `bitpack` options into the class header:

```python
from typing import Annotated

import jax
import jax.numpy as jnp

from xtructure import FieldDescriptor, Xtructurable


class State(Xtructurable, validate=True, bitpack="aggregate"):
    flags: Annotated[jax.Array, FieldDescriptor.tensor(dtype=jnp.bool_, shape=(5,), bits=1)]


state = State.default()
packed = state.packed
restored = packed.as_original()
```

For a bare decorator or an empty decorator call, use `class State(Xtructurable):`.
Defaults remain `validate=False` and `bitpack="auto"`. The supported bitpacking
policies are `"auto"`, `"aggregate"`, `"field"`, and `"off"`.

Constructors, `.default()`, indexing, `.at`, batch reshaping, packing, and IO retain
their behavior. Generated packed and unpacked classes also inherit directly from
`Xtructurable`. Preserve a migrated class's module and name when loading existing
data, because IO metadata records them.

Use `typing.Annotated` to expose actual field types to static analyzers.
Descriptor-only field annotations remain supported. The public `Xtructurable`
import exposes the nominal base and its `Self` return types.

## Runtime checks

The `is_xtructed` marker and the `is_xtructure_dataclass_type` and
`is_xtructure_dataclass_instance` helpers have been removed, along with
`xtructure.core.type_utils`. Use ordinary Python nominal checks:

```python
assert issubclass(State, Xtructurable)
assert isinstance(state, Xtructurable)
```

Use `isinstance(value, type) and issubclass(value, Xtructurable)` when the input
may not be a class. `isinstance(State, Xtructurable)` is false: a class is not an
instance of its base class.

## Inheritance boundaries

- Define each data class directly from `Xtructurable`. Inheriting from an already
  transformed data class raises `TypeError`; share behavior through plain Python
  mixins or compose nested fields with `FieldDescriptor`.
- Instantiate a concrete subclass, not `Xtructurable` itself.

`@base_dataclass` remains supported for plain JAX PyTrees such as container
implementations. See [Core Concepts](core_concepts.md) for field descriptors,
runtime validation, and bitpacking policies.

## Inherited methods and field names

Common methods and properties are implemented on `Xtructurable`, including
construction helpers, indexing, batch operations, comparison, hashing, validation,
and IO. Explicit subclass methods take precedence through ordinary Python method
resolution. The `__str__` and `__post_init__` extension hooks retain their wrapping
behavior: user formatting still supports batched display, and the layout cache is
built after the user post-init hook runs.

Field names must not shadow the base's members or internal state. Names such as
`shape`, `layout`, `hash`, `default`, and `_layout_cache` raise `TypeError` at class
definition. Enabled generated members, such as `packed` or `<field>_unpacked`, also
cannot collide with fields. This restriction does not extend to plain classes
using `@base_dataclass`.

`__eq__` and `__ne__` return scalar JAX boolean arrays for instances of the same
type, and `NotImplemented` for other types. `Xtructurable` classes use dataclass
`eq=False` to inherit these methods and remain unhashable with Python's `hash()`;
the explicit `.hash()` method remains available. Direct `@base_dataclass` usage
keeps its existing `eq=True` default.

`from_tuple` and `to_tuple` have been removed from both declaration paths. Their
names are no longer reserved as fields. `set_unpacked` and `from_unpacked` are
inherited metadata-based helpers; field-specific unpacked properties and
aggregate packed classes are still generated according to the bitpack policy.
