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

import dataclasses
import os
from typing import Any, Type, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from xtructure.core.field_descriptors import FieldDescriptor, get_field_descriptors
from xtructure.core.type_utils import is_xtructure_dataclass_type

from .bitpack import _extract_bits
from .kernels import pack_words_all_pallas, pack_words_all_xla
from .spec import (
    _AggLeafSpec,
    _build_agg_spec,
    _build_word_contrib_tables,
    _ceil_div,
    _compute_word_tail_layout,
    _default_unpack_dtype,
)
from .view import build_unpacked_view_cls

T = TypeVar("T")


def add_aggregate_bitpack(cls: Type[T]) -> Type[T]:
    specs, total_bits = _build_agg_spec(cls)
    words_all_len, stored_words_len, tail_bytes = _compute_word_tail_layout(total_bits)
    tables = _build_word_contrib_tables(specs, words_all_len=words_all_len)

    # Build a Packed class with a uint32 word-stream plus optional uint8 tail.
    packed_name = f"{cls.__name__}Packed"

    # Create class dynamically so user doesn't have to write it.
    Packed = type(packed_name, (), {"__module__": cls.__module__})

    # Attach xtructure annotations: words + tail.
    Packed.__annotations__ = {
        "words": FieldDescriptor.tensor(
            dtype=jnp.uint32, shape=(stored_words_len,), fill_value=0
        ),
        "tail": FieldDescriptor.tensor(
            dtype=jnp.uint8, shape=(tail_bytes,), fill_value=0
        ),
    }

    # Delay import to avoid circular dependency during decorator import graph.
    from xtructure.core.xtructure_decorators import (
        xtructure_dataclass as _xtructure_dataclass,
    )

    Packed = _xtructure_dataclass(Packed)  # type: ignore[assignment]

    # Store spec on class (static python data)
    setattr(Packed, "__agg_specs__", specs)
    setattr(Packed, "__agg_total_bits__", total_bits)
    setattr(Packed, "__agg_words_all_len__", words_all_len)
    setattr(Packed, "__agg_words_len__", stored_words_len)
    setattr(Packed, "__agg_tail_bytes__", tail_bytes)
    setattr(Packed, "__agg_total_values__", int(tables.total_values))
    setattr(Packed, "__agg_word_tables__", tables)
    setattr(Packed, "__agg_unpacked_cls__", cls)

    # Build nested logical-view unpacked classes mirroring the original structure,
    # but using default unpack dtypes (bool/uint8/uint32) and with validate=False.
    UnpackedView, _view_cache = build_unpacked_view_cls(
        cls, default_unpack_dtype=_default_unpack_dtype
    )
    setattr(Packed, "__agg_unpacked_view_cls__", UnpackedView)

    def _parse_backend_env(name: str, default: str) -> str:
        value = os.environ.get(name, default)
        value = value.strip().lower()
        if value in {"auto", "pallas", "xla", "off"}:
            return value
        raise ValueError(f"{name} must be one of: auto, pallas, xla, off")

    def _parse_int_env(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None:
            return default
        value = value.strip().lower()
        if value in {"", "none", "auto"}:
            return default
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must be positive")
        return parsed

    def _pack_backend() -> str:
        choice = _parse_backend_env("XTRUCTURE_AGG_BITPACK_PACK_BACKEND", "auto")
        if choice == "off":
            return "xla"
        if choice in {"pallas", "xla"}:
            return choice
        # auto
        backend = jax.default_backend()
        # For the typical <=1e3 byte packed payloads, the vectorized XLA path is
        # generally faster on GPU, while TPU benefits more consistently from pallas.
        if backend == "tpu":
            return "pallas"
        return "xla"

    def _pack_word_tile() -> int:
        return _parse_int_env("XTRUCTURE_AGG_BITPACK_PACK_WORD_TILE", 4)

    def _pack_instance(instance: Any) -> tuple[jax.Array, jax.Array]:
        # Determine batch shape from the first field (xtructure invariant already implies consistent batching)
        batch = getattr(instance.shape, "batch", ())
        if batch == -1:
            raise TypeError(f"{cls.__name__} is UNSTRUCTURED; cannot aggregate-pack.")

        # Flatten batch dims
        flat_n = int(np.prod(np.array(batch, dtype=np.int64))) if batch else 1

        def _get_path(obj: Any, path: tuple[str, ...]) -> Any:
            out = obj
            for p in path:
                out = getattr(out, p)
            return out

        # Prepare per-leaf flattened arrays of shape (flat_n, nvalues), cast to uint32.
        fields_u32 = []
        for s in specs:
            arr = jnp.asarray(_get_path(instance, s.path))
            arr_flat = arr.reshape((flat_n, s.nvalues)).astype(jnp.uint32)
            fields_u32.append(arr_flat)

        total_values = int(tables.total_values)
        if total_values == 0:
            words_all_2d = jnp.zeros((flat_n, words_all_len), dtype=jnp.uint32)
        else:
            values_stream = (
                jnp.concatenate(fields_u32, axis=1)
                if len(fields_u32) > 1
                else fields_u32[0]
            )
            if values_stream.shape[1] != total_values:
                raise ValueError(
                    "aggregate_bitpack internal error: total_values mismatch"
                )

            backend_choice = _pack_backend()
            if backend_choice == "pallas":
                pallas_backend = (
                    "triton" if jax.default_backend() == "gpu" else "mosaic_tpu"
                )
                words_all_2d = pack_words_all_pallas(
                    values_stream,
                    tables,
                    backend=pallas_backend,
                    word_tile=_pack_word_tile(),
                )
            else:
                words_all_2d = pack_words_all_xla(values_stream, tables)

        if tail_bytes == 0:
            stored_words_2d = words_all_2d
            tail_2d = jnp.zeros((flat_n, 0), dtype=jnp.uint8)
        else:
            last = words_all_2d[:, -1]
            stored_words_2d = (
                words_all_2d[:, :-1]
                if words_all_len > 1
                else jnp.zeros((flat_n, 0), jnp.uint32)
            )
            shifts = jnp.arange(tail_bytes, dtype=jnp.uint32) * jnp.uint32(8)
            tail_2d = ((last[:, None] >> shifts[None, :]) & jnp.uint32(0xFF)).astype(
                jnp.uint8
            )

        packed_words = stored_words_2d.reshape(batch + (stored_words_len,))
        packed_tail = (
            tail_2d.reshape(batch + (tail_bytes,))
            if tail_bytes
            else jnp.zeros(batch + (0,), jnp.uint8)
        )
        return packed_words, packed_tail

    def packed_prop(self):
        words, tail = _pack_instance(self)
        return Packed(words=words, tail=tail)  # type: ignore[call-arg]

    setattr(cls, "Packed", Packed)
    setattr(cls, "packed", property(packed_prop))

    def bitpack_schema(cls_):
        """Return a plain-Python description of the aggregate bitpacking layout."""
        storage_bytes = int(stored_words_len * 4 + tail_bytes)
        payload_bytes = int(_ceil_div(total_bits, 8))
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
        words = jnp.asarray(packed.words, dtype=jnp.uint32).reshape(
            (flat_n, stored_words_len)
        )
        if tail_bytes == 0:
            return words

        tail = jnp.asarray(packed.tail, dtype=jnp.uint8).reshape((flat_n, tail_bytes))
        shifts = jnp.arange(tail_bytes, dtype=jnp.uint32) * jnp.uint32(8)
        last_parts = tail.astype(jnp.uint32) << shifts[None, :]
        last = jnp.bitwise_or.reduce(last_parts, axis=1)
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

    def _decode_field(row_words: jax.Array, s: _AggLeafSpec, indices: Any) -> jax.Array:
        """Decode selected flattened indices for one field from a single row of words."""
        idxs = _normalize_indices(indices, nvalues=s.nvalues)
        # Convert to bit positions and extract bits per index.
        bit_pos = jnp.uint32(s.bit_offset) + idxs.astype(jnp.uint32) * jnp.uint32(
            s.bits
        )

        # Vectorized extraction (avoids per-element vmap).
        word_idx = jnp.right_shift(bit_pos, jnp.uint32(5)).astype(jnp.int32)
        shift = (bit_pos & jnp.uint32(31)).astype(jnp.uint32)

        w0 = row_words[word_idx]
        w1 = jnp.take(row_words, word_idx + 1, mode="fill")

        low = jnp.right_shift(w0, shift)
        high = jnp.where(
            shift == 0, jnp.uint32(0), jnp.left_shift(w1, jnp.uint32(32) - shift)
        )

        mask = jnp.uint32(0xFFFFFFFF) if s.bits == 32 else jnp.uint32((1 << s.bits) - 1)
        vals = jnp.bitwise_and(jnp.bitwise_or(low, high), mask).astype(jnp.uint32)
        if s.bits == 1:
            if s.unpack_dtype == jnp.bool_:
                return vals.astype(jnp.bool_)
            return vals.astype(s.unpack_dtype)
        return vals.astype(s.unpack_dtype)

    def _decode_field_all(
        words_all: jax.Array, s: _AggLeafSpec, indices: Any
    ) -> jax.Array:
        """Decode selected flattened indices for one field for all rows.

        Args:
            words_all: (flat_n, words_all_len) uint32.
        Returns:
            (flat_n, nvalues_selected) array with dtype per s.unpack_dtype.
        """
        idxs = _normalize_indices(indices, nvalues=s.nvalues)
        bit_pos = jnp.uint32(s.bit_offset) + idxs.astype(jnp.uint32) * jnp.uint32(
            s.bits
        )

        word_idx = jnp.right_shift(bit_pos, jnp.uint32(5)).astype(jnp.int32)
        shift = (bit_pos & jnp.uint32(31)).astype(jnp.uint32)

        flat_n = int(words_all.shape[0])
        words_padded = jnp.concatenate(
            [words_all, jnp.zeros((flat_n, 1), dtype=jnp.uint32)], axis=1
        )

        w0 = jnp.take(words_padded, word_idx, axis=1)
        w1 = jnp.take(words_padded, word_idx + 1, axis=1)

        shift2d = shift[None, :]
        low = jnp.right_shift(w0, shift2d)
        high = jnp.where(
            shift2d == 0,
            jnp.uint32(0),
            jnp.left_shift(w1, jnp.uint32(32) - shift2d),
        )

        mask = jnp.uint32(0xFFFFFFFF) if s.bits == 32 else jnp.uint32((1 << s.bits) - 1)
        vals = jnp.bitwise_and(jnp.bitwise_or(low, high), mask).astype(jnp.uint32)

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
            decoded = _decode_field_all(words_all, s, None)
            decoded = decoded.reshape(batch + s.unpacked_shape)
            decoded_by_path[s.path] = decoded

        def _build_view_instance(
            orig: type, view: type, prefix: tuple[str, ...]
        ) -> Any:
            descs = get_field_descriptors(orig)
            kwargs = {}
            for field in dataclasses.fields(orig):
                name = field.name
                fd = descs.get(name)
                if fd is None:
                    continue
                if is_xtructure_dataclass_type(fd.dtype):
                    nested_view = _view_cache[fd.dtype]
                    kwargs[name] = _build_view_instance(
                        fd.dtype, nested_view, prefix + (name,)
                    )
                else:
                    kwargs[name] = decoded_by_path[prefix + (name,)]
            return view(**kwargs)

        return _build_view_instance(cls, UnpackedView, ())

    setattr(Packed, "unpacked", property(unpacked_prop))
    setattr(Packed, "bitpack_schema", classmethod(lambda cls_: cls.bitpack_schema()))  # type: ignore[misc]

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
            raise ValueError(
                f"dtype_policy must be 'default' or 'declared', got {dtype_policy!r}"
            )

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

        decoded = _decode_field_all(words_all, spec, indices)

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
            raise ValueError(
                f"dtype_policy must be 'default' or 'declared', got {dtype_policy!r}"
            )
        if dtype_policy == "default":
            return self.unpacked

        u = self.unpacked

        def _to_declared(orig: type, view_obj: Any) -> Any:
            descs = get_field_descriptors(orig)
            kwargs = {}
            for field in dataclasses.fields(orig):
                name = field.name
                fd = descs.get(name)
                if fd is None:
                    continue
                val = getattr(view_obj, name)
                if is_xtructure_dataclass_type(fd.dtype):
                    kwargs[name] = _to_declared(fd.dtype, val)
                else:
                    try:
                        kwargs[name] = jnp.asarray(val).astype(fd.dtype)
                    except TypeError:
                        kwargs[name] = val
            return orig(**kwargs)

        return _to_declared(cls, u)

    def as_original(self):
        """Backward-compatible alias for `unpack(dtype_policy="declared")`."""
        return self.unpack(dtype_policy="declared")

    setattr(Packed, "as_original", as_original)
    setattr(Packed, "unpack", unpack)
    return cls
