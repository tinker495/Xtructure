"""Aggregate bitpacking across fields of a dataclass.

This decorator adds a `.packed` property that returns a packed representation
containing a word-aligned `uint32` stream plus an optional `uint8` tail, and a
`.unpacked` property on the packed representation that reconstructs a logical view.

Opt-in via `@xtructure_dataclass(aggregate_bitpack=True)`.

Rules:
- Only pack primitive array-like fields whose FieldDescriptor.bits is set.
- `bits` can be 1..32.
- Nested xtructure_dataclass fields are supported for scalar nested fields (intrinsic_shape == ()).
"""

from __future__ import annotations

from typing import Any, Type, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from xtructure.core.field_descriptors import FieldDescriptor
from xtructure.core.layout import get_type_layout
from xtructure.core.layout.bitpack import ceil_div
from xtructure.core.layout.traversal import (
    build_instance_from_leaf_values,
    get_path_value,
    iter_leaf_values,
)
from xtructure.core.layout.types import AggregateBitpackReason, AggregateLeafLayout

from .bits import _extract_bits, _insert_bits
from .generated import GENERATED_PACKED_ROLE, register_generated_class
from .view import build_unpacked_view_cls

T = TypeVar("T")


def add_aggregate_bitpack(cls: Type[T]) -> Type[T]:
    aggregate = get_type_layout(cls).aggregate_bitpack
    if not aggregate.eligible:
        reason = aggregate.reason or "aggregate_bitpack is not eligible for this type."
        if aggregate.reason_kind is AggregateBitpackReason.SCALAR_NESTED:
            raise NotImplementedError(reason)
        raise ValueError(reason)
    specs = list(aggregate.leaves)
    total_bits = int(aggregate.total_bits)
    words_all_len = aggregate.words_all_len
    stored_words_len = aggregate.stored_words_len
    tail_bytes = aggregate.tail_bytes

    # Build a Packed class with a uint32 word-stream plus optional uint8 tail.
    packed_name = f"{cls.__name__}Packed"

    # Create class dynamically so user doesn't have to write it.
    Packed = type(packed_name, (), {"__module__": cls.__module__})

    # Attach xtructure annotations: words + tail.
    Packed.__annotations__ = {
        "words": FieldDescriptor.tensor(dtype=jnp.uint32, shape=(stored_words_len,), fill_value=0),
        "tail": FieldDescriptor.tensor(dtype=jnp.uint8, shape=(tail_bytes,), fill_value=0),
    }

    # Delay import to avoid circular dependency during decorator import graph.
    from xtructure.core.xtructure_decorators import (
        xtructure_dataclass as _xtructure_dataclass,
    )

    Packed = _xtructure_dataclass(Packed)  # type: ignore[assignment]

    register_generated_class(Packed, role=GENERATED_PACKED_ROLE)

    # Build nested logical-view unpacked classes mirroring the original structure,
    # but using default unpack dtypes (bool/uint8/uint32) and with validate=False.
    UnpackedView, _view_cache = build_unpacked_view_cls(cls)

    def _pack_instance(instance: Any) -> tuple[jax.Array, jax.Array]:
        # Determine batch shape from the first field (xtructure invariant already implies consistent batching)
        batch = getattr(instance.shape, "batch", ())
        if batch == -1:
            raise TypeError(f"{cls.__name__} is UNSTRUCTURED; cannot aggregate-pack.")

        # Flatten batch dims
        flat_n = int(np.prod(np.array(batch, dtype=np.int64))) if batch else 1

        # Prepare per-leaf flattened arrays of shape (flat_n, nvalues)
        field_rows = []
        for s in specs:
            arr = jnp.asarray(get_path_value(instance, s.path))
            arr_flat = arr.reshape((flat_n, s.nvalues))
            field_rows.append(arr_flat)

        def _pack_row(*row_fields):
            # Pack into full uint32 word stream of length words_all_len.
            words = jnp.zeros((words_all_len,), dtype=jnp.uint32)

            for s, values in zip(specs, row_fields):
                vals_u32 = values.astype(jnp.uint32)

                def body(i, w):
                    bit_pos = jnp.uint32(s.bit_offset) + jnp.uint32(i) * jnp.uint32(s.bits)
                    v = vals_u32[i]
                    return _insert_bits(w, bit_pos, v, s.bits)

                words = lax.fori_loop(0, s.nvalues, body, words)

            if tail_bytes == 0:
                stored_words = words
                tail = jnp.zeros((0,), dtype=jnp.uint8)
                return stored_words, tail

            # Split last word into 1..2 byte tail.
            last = words[-1]
            tail = jnp.stack(
                [
                    ((last >> jnp.uint32(8 * i)) & jnp.uint32(0xFF)).astype(jnp.uint8)
                    for i in range(tail_bytes)
                ]
            )
            stored_words = words[:-1] if words_all_len > 1 else jnp.zeros((0,), dtype=jnp.uint32)
            return stored_words, tail

        packed_words_2d, packed_tail_2d = jax.vmap(_pack_row)(*field_rows)
        packed_words = packed_words_2d.reshape(batch + (stored_words_len,))
        packed_tail = (
            packed_tail_2d.reshape(batch + (tail_bytes,))
            if tail_bytes
            else jnp.zeros(batch + (0,), dtype=jnp.uint8)
        )
        return packed_words, packed_tail

    def packed_prop(self):
        words, tail = _pack_instance(self)
        return Packed(words=words, tail=tail)

    setattr(cls, "Packed", Packed)
    setattr(cls, "packed", property(packed_prop))

    def bitpack_schema(cls_):
        """Return a plain-Python description of the aggregate bitpacking layout."""
        storage_bytes = int(stored_words_len * 4 + tail_bytes)
        payload_bytes = int(ceil_div(total_bits, 8))
        return {
            "mode": "aggregate",
            "class": f"{cls.__module__}.{cls.__name__}",
            "total_bits": int(total_bits),
            "payload_bytes": payload_bytes,
            "storage_bytes": storage_bytes,
            "words_all_len": int(words_all_len),
            "words_len": int(stored_words_len),
            "tail_bytes": int(tail_bytes),
            "fields": [
                {
                    "path": ".".join(s.path),
                    "bits": int(s.bits),
                    "bit_offset": int(s.bit_offset),
                    "bit_len": int(s.bit_len),
                    "nvalues": int(s.nvalues),
                    "unpacked_shape": tuple(s.unpacked_shape),
                    "unpacked_dtype_default": str(jnp.dtype(s.unpack_dtype)),
                    "declared_dtype": str(jnp.dtype(s.declared_dtype)),
                }
                for s in specs
            ],
        }

    setattr(cls, "bitpack_schema", classmethod(bitpack_schema))

    def _words_all_from_packed(packed: Any) -> jax.Array:
        """Return (flat_n, words_all_len) uint32 words, reconstructing last word from tail if needed."""
        batch = getattr(packed.shape, "batch", ())
        if batch == -1:
            raise TypeError(f"{packed_name} is UNSTRUCTURED; cannot unpack.")
        flat_n = int(np.prod(np.array(batch, dtype=np.int64))) if batch else 1
        words = jnp.asarray(packed.words, dtype=jnp.uint32).reshape((flat_n, stored_words_len))
        if tail_bytes == 0:
            return words

        tail = jnp.asarray(packed.tail, dtype=jnp.uint8).reshape((flat_n, tail_bytes))
        last = jnp.uint32(0)
        for i in range(tail_bytes):
            last = last | (tail[:, i].astype(jnp.uint32) << jnp.uint32(8 * i))
        if stored_words_len:
            return jnp.concatenate([words, last[:, None]], axis=1)
        return last[:, None]

    def _normalize_indices(indices: Any, *, nvalues: int) -> jax.Array:
        """Normalize indices to a 1D int32 JAX array."""
        if indices is None:
            return jnp.arange(nvalues, dtype=jnp.int32)
        if isinstance(indices, slice):
            start = 0 if indices.start is None else int(indices.start)
            stop = nvalues if indices.stop is None else int(indices.stop)
            step = 1 if indices.step is None else int(indices.step)
            return jnp.arange(start, stop, step, dtype=jnp.int32)
        if isinstance(indices, (list, tuple, np.ndarray)):
            return jnp.asarray(indices, dtype=jnp.int32).reshape((-1,))
        return jnp.asarray(indices, dtype=jnp.int32).reshape((-1,))

    def _decode_field(row_words: jax.Array, s: AggregateLeafLayout, indices: Any) -> jax.Array:
        """Decode selected flattened indices for one field from a single row of words."""
        idxs = _normalize_indices(indices, nvalues=s.nvalues)
        # Convert to bit positions and extract bits per index.
        bit_pos = jnp.uint32(s.bit_offset) + idxs.astype(jnp.uint32) * jnp.uint32(s.bits)
        vals = jax.vmap(lambda bp: _extract_bits(row_words, bp, s.bits))(bit_pos).astype(jnp.uint32)
        if s.bits == 1:
            if s.unpack_dtype == jnp.bool_:
                return vals.astype(jnp.bool_)
            return vals.astype(s.unpack_dtype)
        return vals.astype(s.unpack_dtype)

    # Add unpacking on Packed (full unpack).
    def unpacked_prop(self):
        batch = getattr(self.shape, "batch", ())
        if batch == -1:
            raise TypeError(f"{packed_name} is UNSTRUCTURED; cannot unpack.")
        words_all = _words_all_from_packed(self)

        # Decode each leaf spec in a memory-efficient way (one leaf at a time),
        # then reconstruct nested view instances.
        decoded_by_path: dict[tuple[str, ...], jax.Array] = {}
        for s in specs:
            decoded = jax.vmap(lambda row: _decode_field(row, s, None))(
                words_all
            )  # (flat_n, nvalues)
            decoded = decoded.reshape(batch + s.unpacked_shape)
            decoded_by_path[s.path] = decoded

        return build_instance_from_leaf_values(
            cls,
            decoded_by_path,
            type_map=_view_cache,
        )

    setattr(Packed, "unpacked", property(unpacked_prop))

    def packed_bitpack_schema(_packed_cls):
        # Delegate to the original (unpacked) class so the schema is authored once.
        return cls.bitpack_schema()

    setattr(Packed, "bitpack_schema", classmethod(packed_bitpack_schema))

    def unpack_field(
        self,
        name: str,
        *,
        indices: Any = None,
        dtype_policy: str = "default",
    ):
        """Decode a single field from the aggregated packed buffer.

        This avoids materializing the full `.unpacked` dataclass.

        Args:
            name: Field name to decode.
            indices: Optional indices into the *flattened* field values (0..nvalues-1).
                Can be None (decode all), a Python slice, or an int/array/list of ints.
                Returned shape:
                - None: batch + unpacked_shape
                - indices provided: batch + (len(indices),)
            dtype_policy:
                - "default": decode to default dtype (bool/uint8/uint32 based on bits)
                - "declared": decode then cast to the field's declared dtype
        """
        if dtype_policy not in ("default", "declared"):
            raise ValueError(f"dtype_policy must be 'default' or 'declared', got {dtype_policy!r}")

        # Support dotted paths for nested leaves: "inner.codes"
        path = tuple(name.split(".")) if isinstance(name, str) else tuple(name)
        spec = None
        for s in specs:
            if s.path == path:
                spec = s
                break
        if spec is None:
            raise KeyError(f"Unknown field '{name}' for {packed_name}.")

        batch = getattr(self.shape, "batch", ())
        if batch == -1:
            raise TypeError(f"{packed_name} is UNSTRUCTURED; cannot unpack_field.")
        words_all = _words_all_from_packed(self)

        decoded = jax.vmap(lambda row: _decode_field(row, spec, indices))(words_all)

        # Shape the output.
        flat_n = decoded.shape[0]
        batch_size = int(np.prod(np.array(batch, dtype=np.int64))) if batch else 1
        assert flat_n == batch_size

        if indices is None:
            out = decoded.reshape(batch + spec.unpacked_shape)
        else:
            out = decoded.reshape(batch + (decoded.shape[1],))

        if dtype_policy == "declared":
            try:
                out = out.astype(spec.declared_dtype)
            except TypeError:
                pass
        return out

    setattr(Packed, "unpack_field", unpack_field)

    def unpack(self, *, dtype_policy: str = "default"):
        """Unpack the aggregated byte-stream.

        Args:
            dtype_policy:
                - "default": return the logical view type (`<Cls>Unpacked`).
                  Dtypes default to bool for 1-bit, uint8 for <=8, uint32 for >8.
                - "declared": return the original dataclass type, casting fields
                  back to their declared dtypes (validation-friendly).
        """
        if dtype_policy not in ("default", "declared"):
            raise ValueError(f"dtype_policy must be 'default' or 'declared', got {dtype_policy!r}")
        if dtype_policy == "default":
            return self.unpacked

        view_values = {leaf.path: value for leaf, value in iter_leaf_values(self.unpacked)}
        return build_instance_from_leaf_values(cls, view_values, cast_declared=True)

    def as_original(self):
        """Backward-compatible alias for `unpack(dtype_policy="declared")`."""
        return self.unpack(dtype_policy="declared")

    setattr(Packed, "as_original", as_original)
    setattr(Packed, "unpack", unpack)
    return cls
