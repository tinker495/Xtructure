# Legacy Decorator Migration

`@xtructure_dataclass` is a legacy compatibility API. Applying it emits a
`FutureWarning` at the caller's class definition or explicit decoration call.
This includes the bare decorator, `@xtructure_dataclass()`, and decorators with
`validate` or `bitpack` options. The root `xtructure` export, `xtructure.core`
export, and implementation-module import all use the same warning behavior.
Importing the decorator alone does not warn.

The warning is visible under Python's default warning filters. Existing decorated
classes still work when warnings are allowed. There is no scheduled removal
version in this migration phase.

## Replace the decorator with inheritance

Before (legacy; emits `FutureWarning`):

```python
from typing import Annotated

import jax
import jax.numpy as jnp

from xtructure import FieldDescriptor, xtructure_dataclass


@xtructure_dataclass(validate=True, bitpack="aggregate")
class State:
    flags: Annotated[jax.Array, FieldDescriptor.tensor(dtype=jnp.bool_, shape=(5,), bits=1)]
```

After:

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

Remove the decorator and its import, add `Xtructurable` as a base, and move the
same options into the class header. For the bare decorator or an empty decorator
call, use `class State(Xtructurable):`; defaults remain `validate=False` and
`bitpack="auto"`. The supported bitpacking policies are `"auto"`, `"aggregate"`,
`"field"`, and `"off"`.

Constructors, `.default()`, indexing, `.at`, batch reshaping, packing, and IO use
the shared transformation. Inheritance and generated packed/unpacked classes do
not emit the legacy warning. Keep the class's module and name when migrating
persisted data, because IO metadata records them.

Use `typing.Annotated` to expose actual field types to static analyzers. The
`from xtructure import Xtructurable` public import exposes the nominal base and
its `Self` return types to static analyzers as well as runtime callers.
Descriptor-only field annotations remain supported; they are separate from the
deprecated decorator.
Legacy decorated classes retain runtime compatibility checks, but a decorator
does not make them nominal subclasses for static typing.

## Treat legacy usage as an error

To reject legacy declarations when starting a Python application:

```bash
python -W error::FutureWarning your_application.py
```

This promotes all `FutureWarning` categories to errors, including warnings from
dependencies. For Xtructure decorator warnings only, install a message filter
before importing modules that define legacy classes:

```python
import warnings

warnings.filterwarnings("error", message=r"@xtructure_dataclass\b", category=FutureWarning)

import your_application
```

The exception occurs before dataclass conversion or JAX PyTree registration, so
the rejected class is not partially transformed. A module that defines legacy
classes fails to import under this policy until those classes are migrated.

## Invalid combinations already raise errors

- Do not decorate an `Xtructurable` subclass: class creation has already applied
  the transformation, so the decorator raises `TypeError`.
- Do not apply `@xtructure_dataclass` twice: repeated transformation raises
  `TypeError`.
- Subclassing an `Xtructurable` data class, or combining `Xtructurable` with a
  legacy decorated base, raises `TypeError`. Define each data class directly
  from `Xtructurable`. Share behavior through plain Python mixins or nest
  structures with `FieldDescriptor` instead.
- Do not instantiate `Xtructurable` directly; instantiate a concrete subclass.

The `@base_dataclass` decorator remains supported for plain JAX PyTrees such as
container implementations. It does not emit this legacy warning. See
[Core Concepts](core_concepts.md) for field descriptors, runtime validation, and
bitpacking policies.
