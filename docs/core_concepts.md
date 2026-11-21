# Core Concepts: Defining Custom Data Structures

Before using `HashTable` or `BGPQ` in Xtructure, define your data schema by inheriting from `Xtructurable` and declaring fields with `FieldDescriptor`.

```python
import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable


class MyDataValue(Xtructurable):
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    position: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(4,))


class MyHeapItem(Xtructurable):
    task_id: FieldDescriptor.scalar(dtype=jnp.int32)
    payload: FieldDescriptor.tensor(dtype=jnp.float32, shape=(2, 2))
```

## `Xtructurable`

Inheriting from `Xtructurable` registers a class as a JAX PyTree backed by Structure-of-Arrays (SoA) leaf arrays while exposing Array-of-Structures (AoS) semantics. Class creation computes layout metadata, compiles field updaters, and optionally generates packed representations. Explicit subclass methods take precedence, except `__str__` and `__post_init__` remain wrapped extension hooks. Field names colliding with the base API or generated members raise `TypeError` at class definition.

Define each structured class directly from `Xtructurable` (subclassing an already-xtructured class is disallowed; share behavior via plain Python mixins or compose nested fields via `FieldDescriptor`). Use `isinstance(value, Xtructurable)` / `issubclass(cls, Xtructurable)`, JAX boolean array equality (`==` / `!=`), and `.hash()` for hashing (instances do not support Python's `hash()`).

Key interface contracts:

*   **Shape and layout metadata**: `.shape` returns the leading batch axes shared by all fields (`()` for `SINGLE`, `(N, ...)` for `BATCHED`), `.structured_type` returns `StructuredType.SINGLE` or `StructuredType.BATCHED`, `.layout.field_shape_for("field")` gives a field's intrinsic shape, and `.default_shape` / `.dtype` report unbatched shapes and declared dtypes.
*   **Construction**: `cls.default(shape=())` and `cls.random(shape=(), key=...)` prepend batch axes `shape` to each field's intrinsic shape.
*   **Indexing and iteration**: `instance[idx]` indexes/slices every field uniformly. `len(instance)` returns the first batch dimension for `BATCHED` (`1` for `SINGLE`). Iterating (`for item in instance`) yields elements along the first batch axis and stops cleanly at the batch boundary even though JAX array indexing clamps out-of-bounds indices; iterating a `SINGLE` instance raises `TypeError`.
*   **Batch transformations and updates**: `reshape` and `flatten` operate on batch dimensions of `BATCHED` instances. `at[idx].set(...)` and `at[idx].set_as_condition(cond, ...)` return updated copies out-of-place.
*   **Display**: `str(instance)` / `instance.str(**kwargs)` uses custom `__str__(self, **kwargs)` (or `_custom_pretty_formatter`) for `SINGLE`, and `BatchedRenderer` plain-text column layout for `BATCHED` (truncated via `MAX_PRINT_BATCH_SIZE` and `SHOW_BATCH_SIZE`).
*   **Serialization**: `instance.save(path, packed=True)` and `cls.load(path)` serialize to `.npz`. Treat `.npz` files as trusted inputs: loading imports the module recorded in file metadata to reconstruct the class.

## `FieldDescriptor`

`FieldDescriptor` declares a field's JAX dtype (or nested `Xtructurable` subclass), intrinsic (non-batch) shape, default fill policy (`fill_value` or `fill_value_factory`), optional `bits` width (`[1, 32]` for aggregate bitpacking and `save(..., packed=True)`), and optional `validator`.

*   Factory methods (recommended): `FieldDescriptor.scalar(dtype=..., default=..., ...)` and `FieldDescriptor.tensor(dtype=..., shape=..., fill_value=..., ...)`.
*   Shorthand syntax: `FieldDescriptor[jax_dtype]`, `FieldDescriptor[jax_dtype, shape_tuple]`, or `FieldDescriptor(dtype=..., intrinsic_shape=..., ...)`.
*   Default `fill_value` when omitted: max representable value for unsigned ints, `jnp.inf` for signed ints and floats, and recursive `.default()` for nested structures.

```python
def require_unit_interval(x):
    if jnp.any((x < 0.0) | (x > 1.0)):
        raise ValueError("Must be probability in [0, 1]")


class BetterSyntax(Xtructurable):
    count: FieldDescriptor.scalar(dtype=jnp.int32, default=0)
    points: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,), fill_value=0.0)
    prob: FieldDescriptor.scalar(dtype=jnp.float32, validator=require_unit_interval)
```

### `Annotated` syntax

`typing.Annotated` exposes the runtime array type to static analyzers while keeping identical runtime descriptor metadata:

```python
from typing import Annotated
import jax
import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable


class ShorthandSyntax(Xtructurable):
    value: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


class AnnotatedSyntax(Xtructurable):
    value: Annotated[jax.Array, FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))]
```

### Nested structures and `fill_value_factory`

Descriptors can reference another `Xtructurable` subclass so batch shapes and `.default()` / `.random()` flow recursively. When a sentinel depends on the concrete batch shape, pass `fill_value_factory(field_shape, dtype)` instead of a scalar `fill_value`:

```python
class SimpleData(Xtructurable):
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    value: FieldDescriptor.scalar(dtype=jnp.float32)


def nan_fill(field_shape, dtype):
    return jnp.full(field_shape, jnp.nan, dtype=dtype)


class Container(Xtructurable):
    simple: FieldDescriptor.scalar(dtype=SimpleData)
    metrics: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,), fill_value_factory=nan_fill)


instance = Container.default(shape=(8,))
assert instance.simple.value.shape == (8,)
```

### Runtime validation

Every ordinary constructor validates field dtypes, trailing intrinsic shapes, nested types, shared batch axes, and custom validators after user `__post_init__` logic. The same contract applies to `replace`, indexing/updates, `xnp` operations, and file or pickle restoration—operations validate both inputs and results without implicit dtype casting. Wrong dtypes raise `TypeError`; incompatible intrinsic shapes or batch axes raise `ValueError`. `UNSTRUCTURED` is an internal shape-mismatch diagnostic, not a usable data state.

```python
def require_positive(value):
    if jnp.any(value <= 0):
        raise ValueError("Must be positive")


class StrictData(Xtructurable):
    vector: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))
    positive: FieldDescriptor.scalar(dtype=jnp.int32, validator=require_positive)


StrictData(
    vector=jnp.ones((5, 3), dtype=jnp.float32),
    positive=jnp.full((5,), 10, dtype=jnp.int32),
)
```

JAX `tree_map` and `tree_unflatten` restore structure without running user `__post_init__` or validation so axis specifications, abstract tracers, and intermediate transformation trees remain valid under `jit`, `vmap`, and differentiation. The first Xtructure data API call validates a reconstructed object and caches its layout; plain attribute access and JAX PyTree traversal remain structural operations.

Validators execute at data boundaries including JIT tracing, so validators used inside compiled code must be tracer-compatible (a value-dependent Python `if` like `require_positive` raises a JAX tracing error rather than being silently skipped). Validated layout queries reuse the layout cache; call `instance.check_invariants()` to re-run all checks and refresh the cache explicitly.

Always use `replace(...)` for mutations so validation and layout consistency hold; direct attribute reassignment or custom overrides of data methods bypass tracking. User `__post_init__` runs on construction, `replace`, and pickle restoration before invariant checks execute. Generated packed classes and unpacked views validate their own schemas, and original validators run when `as_original()` or declared-dtype restoration constructs the original class.

## In-memory bitpacked fields (memory-optimized runtime state)

Use `FieldDescriptor.packed_tensor(...)` to declare a field's logical shape while storing it in memory as a packed `uint8` byte stream:

- Access `<field>_unpacked` to decode the logical array.
- Use `set_unpacked(field=...)` to pack a logical array into an updated copy.
- Use `from_unpacked(...)` to construct a packed instance directly from logical arrays without an intermediate `.default()`. Ordinary fields passed alongside packed fields must already match their declared dtype.
- Query `get_type_layout(cls)` to inspect derived storage byte counts rather than reading `FieldDescriptor`.

```python
import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable


class PuzzleState(Xtructurable):
    faces: FieldDescriptor.packed_tensor(shape=(6, 54), packed_bits=3)


state = PuzzleState.default()
faces = state.faces_unpacked  # shape (6, 54), dtype uint8
state2 = state.set_unpacked(faces=(faces + 1) & jnp.uint8(7))
packed = PuzzleState.from_unpacked(faces=faces)
```

Notes:
- `packed_bits` supports `[1, 32]`.
- `shape=` specifies the logical unpacked shape; the stored field shape is the derived byte-stream shape.
- When `unpacked_dtype` is omitted, the default unpacked dtype is `bool_` for `packed_bits == 1`, `uint8` for `2 <= packed_bits <= 8`, and `uint32` for `packed_bits > 8`.

## Aggregate bitpacking across all fields (single byte-stream per instance)

Aggregate bitpacking packs all primitive leaves together into a single contiguous bitstream (`words: uint32[...]`, `tail: uint8[0..2]`) to eliminate per-field byte padding.

Set `bitpack=...` in the class header (`class State(Xtructurable, bitpack="aggregate"):`):

- `bitpack="auto"` (default): enables aggregate packing automatically when every primitive leaf (including scalar nested `Xtructurable` fields) declares `bits=...`, and enables `packed_tensor` accessors.
- `bitpack="aggregate"`: requires aggregate packing for the whole dataclass.
- `bitpack="field"`: enables only `FieldDescriptor.packed_tensor(...)` accessors without aggregate packing.
- `bitpack="off"`: disables in-memory bitpacking helpers on the class.

```python
import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable


class AggState(Xtructurable):
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(17,), bits=1)
    faces: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(6, 54), bits=3)
    codes: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(5,), bits=12)


p = AggState.Packed.default()
u = p.unpacked  # AggStateUnpacked view (default unpacked dtypes: bool/uint8/uint32)
o = p.as_original()  # AggState reconstructed and validated with declared dtypes

faces_only = p.unpack_field("faces")
codes_some = p.unpack_field("codes", indices=[0, 2])
p2 = AggState.default().packed
```

Aggregate bitpacking reserves module-level names so IO metadata can round-trip generated classes:

- `<Name>Packed` (e.g. `AggStatePacked`) backs `<Name>.Packed`.
- `<Name>Unpacked` (and `<NestedName>Unpacked` for nested views) backs `.unpacked`.
- Reloading or redefining the same aggregate class replaces earlier generated classes with the same generated role; user-defined module symbols with these reserved names raise `TypeError` during class definition.

## `base_dataclass` and `static_fields`

Plain JAX PyTrees that do not use SoA `FieldDescriptor` fields (such as `HashTable`, `BGPQ`, `Queue`, and `Stack`) use `@base_dataclass`.

`@base_dataclass(static_fields=(...))` marks configuration fields as static PyTree `aux_data` instead of dynamic JAX array leaves. Static fields stay on host as compile-time constants inside `jax.jit` (and participate in JIT cache keys) so sizes, capacities, and branching factors do not need to be threaded as explicit `static_argnums` or re-inferred from array shapes. Static field values must be Python-hashable (`int`, `str`, `tuple`, etc.).

```python
import jax
from xtructure import base_dataclass


@base_dataclass(static_fields=("bucket_size",))
class AlgoConfig:
    bucket_size: int
    data: jax.Array


@jax.jit
def f(cfg: AlgoConfig):
    return cfg.data.reshape((-1, int(cfg.bucket_size)))
```

## SoA storage with AoS ergonomics

See [Structure Layout Flexibility](./layout_flexibility.md) for how Xtructure pairs Structure-of-Arrays storage with Array-of-Structures ergonomics.
