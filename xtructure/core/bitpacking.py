from __future__ import annotations

import dataclasses
import json
from typing import Any, Iterable, Type, TypeVar

import chex
import jax
import jax.numpy as jnp
import numpy as np

from xtructure.core.field_descriptors import FieldDescriptor, get_field_descriptors
from xtructure.core.type_utils import is_xtructure_dataclass_type

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class PackedFieldSpec:
    """A single packed leaf entry.

    - path: dotted field path (e.g. "foo.bar")
    - kind: "bits" or "raw"
    - dtype: numpy dtype string (e.g. "uint8", "float32", "bool")
    - shape: full array shape (including batch dims)
    - bits: only for kind="bits"
    - nbytes: number of payload bytes for this entry
    """

    path: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    bits: int | None
    nbytes: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "bits": self.bits,
            "nbytes": self.nbytes,
        }

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> "PackedFieldSpec":
        return PackedFieldSpec(
            path=str(d["path"]),
            kind=str(d["kind"]),
            dtype=str(d["dtype"]),
            shape=tuple(int(x) for x in d["shape"]),
            bits=None if d.get("bits", None) is None else int(d["bits"]),
            nbytes=int(d["nbytes"]),
        )


@dataclasses.dataclass(frozen=True)
class PackedXtructure:
    """Portable packed representation for xtructure dataclass instances."""

    payload: chex.Array  # uint8
    specs: tuple[PackedFieldSpec, ...]
    cls_module: str
    cls_name: str

    def specs_json(self) -> str:
        return json.dumps([s.to_json_dict() for s in self.specs], separators=(",", ":"))

    @staticmethod
    def specs_from_json(specs_json: str) -> tuple[PackedFieldSpec, ...]:
        raw = json.loads(specs_json)
        return tuple(PackedFieldSpec.from_json_dict(x) for x in raw)


def to_uint8(input: chex.Array, active_bits: int = 1) -> chex.Array:
    """Pack integer/bool arrays into uint8, supporting 1-8 bits per value."""
    assert 1 <= active_bits <= 8, f"active_bits must be 1-8, got {active_bits}"

    if active_bits == 1:
        # Handle boolean arrays efficiently
        if input.dtype == jnp.bool_:
            flatten_input = input.reshape((-1,))
            return jnp.packbits(flatten_input, axis=-1, bitorder="little")
        else:
            flatten_input = (input != 0).reshape((-1,))
            return jnp.packbits(flatten_input, axis=-1, bitorder="little")

    if active_bits in (2, 4, 8):
        assert jnp.issubdtype(
            input.dtype, jnp.integer
        ), f"Input must be integer array for active_bits={active_bits} > 1, got dtype={input.dtype}"
        values_flat = input.flatten()
        if active_bits == 8:
            return values_flat.astype(jnp.uint8)
        values_per_byte = 8 // active_bits
        padding_needed = (values_per_byte - (len(values_flat) % values_per_byte)) % values_per_byte
        if padding_needed > 0:
            values_flat = jnp.concatenate(
                [values_flat, jnp.zeros(padding_needed, dtype=values_flat.dtype)]
            )
        grouped_values = values_flat.reshape(-1, values_per_byte)

        def pack_group(group):
            result = jnp.uint8(0)
            for i, val in enumerate(group):
                result = result | (val.astype(jnp.uint8) << (i * active_bits))
            return result

        return jax.vmap(pack_group)(grouped_values)

    # Efficient block-based packing for 3,5,6,7 bits using only uint32
    assert jnp.issubdtype(
        input.dtype, jnp.integer
    ), f"Input must be integer array for active_bits={active_bits} > 1, got dtype={input.dtype}"
    values_flat = input.flatten()
    L = np.lcm(active_bits, 8)  # total bits per block
    num_values_per_block = L // active_bits
    num_bytes_per_block = L // 8
    padding_needed = (num_values_per_block - (len(values_flat) % num_values_per_block)) % num_values_per_block
    if padding_needed > 0:
        values_flat = jnp.concatenate([values_flat, jnp.zeros(padding_needed, dtype=values_flat.dtype)])
    grouped_values = values_flat.reshape(-1, num_values_per_block)

    if L <= 32:

        def pack_block(group):
            acc = jnp.uint32(0)
            for i in range(num_values_per_block):
                acc = acc | (group[i].astype(jnp.uint32) << (i * active_bits))
            return jnp.array([(acc >> (8 * j)) & 0xFF for j in range(num_bytes_per_block)], dtype=jnp.uint8)

        packed_blocks = jax.vmap(pack_block)(grouped_values)
        return packed_blocks.reshape(-1)

    def pack_block(group):
        packed_bytes = jnp.zeros((num_bytes_per_block,), dtype=jnp.uint8)
        acc = jnp.uint32(0)
        bits_in_acc = 0
        byte_idx = 0
        for i in range(num_values_per_block):
            acc = acc | (group[i].astype(jnp.uint32) << bits_in_acc)
            bits_in_acc += active_bits
            while bits_in_acc >= 8:
                packed_bytes = packed_bytes.at[byte_idx].set((acc & jnp.uint32(0xFF)).astype(jnp.uint8))
                acc = acc >> 8
                bits_in_acc -= 8
                byte_idx += 1
        if byte_idx < num_bytes_per_block:
            packed_bytes = packed_bytes.at[byte_idx].set((acc & jnp.uint32(0xFF)).astype(jnp.uint8))
        return packed_bytes

    packed_blocks = jax.vmap(pack_block)(grouped_values)
    return packed_blocks.reshape(-1)


def from_uint8(packed_bytes: chex.Array, target_shape: tuple[int, ...], active_bits: int = 1) -> chex.Array:
    """Unpack a uint8 array back to values with active_bits per element."""
    assert packed_bytes.dtype == jnp.uint8, f"Input must be uint8, got {packed_bytes.dtype}"
    assert 1 <= active_bits <= 8, f"active_bits must be 1-8, got {active_bits}"

    num_target_elements = int(np.prod(target_shape))
    assert num_target_elements > 0, f"num_target_elements={num_target_elements} must be positive"

    if active_bits == 1:
        all_unpacked_bits = jnp.unpackbits(packed_bytes, count=num_target_elements, bitorder="little")
        return all_unpacked_bits.reshape(target_shape).astype(jnp.bool_)

    if active_bits in (2, 4, 8):
        if active_bits == 8:
            assert len(packed_bytes) >= num_target_elements, "Not enough packed data"
            return packed_bytes[:num_target_elements].reshape(target_shape)
        values_per_byte = 8 // active_bits
        mask = (1 << active_bits) - 1

        def unpack_byte(byte_val):
            values = []
            for i in range(values_per_byte):
                val = (byte_val >> (i * active_bits)) & mask
                values.append(val)
            return jnp.array(values)

        unpacked_groups = jax.vmap(unpack_byte)(packed_bytes)
        all_values = unpacked_groups.flatten()
        assert len(all_values) >= num_target_elements, "Not enough unpacked values"
        return all_values[:num_target_elements].reshape(target_shape).astype(jnp.uint8)

    # Efficient block-based unpacking for 3,5,6,7 bits using only uint32
    L = np.lcm(active_bits, 8)
    num_values_per_block = L // active_bits
    num_bytes_per_block = L // 8
    total_blocks = (len(packed_bytes) + num_bytes_per_block - 1) // num_bytes_per_block
    padding_needed = total_blocks * num_bytes_per_block - len(packed_bytes)
    if padding_needed > 0:
        packed_bytes = jnp.concatenate([packed_bytes, jnp.zeros(padding_needed, dtype=packed_bytes.dtype)])
    grouped_bytes = packed_bytes.reshape(-1, num_bytes_per_block)
    mask = (1 << active_bits) - 1

    if L <= 32:

        def unpack_block(byte_group):
            acc = jnp.uint32(0)
            for j in range(num_bytes_per_block):
                acc = acc | (byte_group[j].astype(jnp.uint32) << (8 * j))
            values = [(acc >> (i * active_bits)) & mask for i in range(num_values_per_block)]
            return jnp.array(values, dtype=jnp.uint8)

        unpacked_blocks = jax.vmap(unpack_block)(grouped_bytes)
        all_values = unpacked_blocks.flatten()
        assert len(all_values) >= num_target_elements, "Not enough unpacked values"
        return all_values[:num_target_elements].reshape(target_shape).astype(jnp.uint8)

    def unpack_block(byte_group):
        values = jnp.zeros((num_values_per_block,), dtype=jnp.uint8)
        acc = jnp.uint32(0)
        bits_in_acc = 0
        byte_idx = 0
        for i in range(num_values_per_block):
            while bits_in_acc < active_bits:
                if byte_idx < num_bytes_per_block:
                    acc = acc | (byte_group[byte_idx].astype(jnp.uint32) << bits_in_acc)
                    bits_in_acc += 8
                    byte_idx += 1
            values = values.at[i].set((acc & jnp.uint32(mask)).astype(jnp.uint8))
            acc = acc >> active_bits
            bits_in_acc -= active_bits
        return values

    unpacked_blocks = jax.vmap(unpack_block)(grouped_bytes)
    all_values = unpacked_blocks.flatten()
    assert len(all_values) >= num_target_elements, "Not enough unpacked values"
    return all_values[:num_target_elements].reshape(target_shape).astype(jnp.uint8)


def _dtype_to_str(dtype: Any) -> str:
    return str(np.dtype(dtype))


def _array_to_raw_bytes(arr: chex.Array) -> chex.Array:
    arr = jnp.asarray(arr)
    if arr.dtype == jnp.bool_:
        # bitcast(bool) isn't consistently supported; store as 0/1 bytes.
        return arr.astype(jnp.uint8).reshape(-1)
    return jax.lax.bitcast_convert_type(arr, jnp.uint8).reshape(-1)


def _raw_bytes_to_array(raw: chex.Array, *, dtype_str: str, shape: tuple[int, ...]) -> chex.Array:
    dt = jnp.dtype(dtype_str)
    raw = jnp.asarray(raw, dtype=jnp.uint8).reshape(-1)
    if dt == jnp.bool_:
        return raw.astype(jnp.bool_).reshape(shape)
    if dt.itemsize == 1:
        return raw.astype(dt).reshape(shape)
    # Repack bytes into dtype lanes via bitcast over trailing axis.
    if raw.size % dt.itemsize != 0:
        raise ValueError(f"Raw payload size {raw.size} not divisible by itemsize {dt.itemsize} for dtype={dtype_str}")
    chunks = raw.reshape(-1, dt.itemsize)
    return jax.lax.bitcast_convert_type(chunks, dt).reshape(shape)


def _validate_bits_range(values: chex.Array, bits: int):
    # Optional, best-effort runtime check. Keep it cheap: only check max for unsigned/bool.
    if bits == 1 and values.dtype == jnp.bool_:
        return
    if not jnp.issubdtype(values.dtype, jnp.integer):
        raise TypeError(f"pack_bits requires integer/bool arrays, got dtype={values.dtype}")
    # For correctness we require non-negative and max < 2**bits.
    vmax = jnp.max(values)
    vmin = jnp.min(values)
    if int(vmin) < 0:
        raise ValueError(f"pack_bits requires non-negative values, got min={int(vmin)}")
    limit = (1 << bits) - 1
    if int(vmax) > limit:
        raise ValueError(f"pack_bits={bits} cannot represent max={int(vmax)} (limit={limit})")


def pack_instance(instance: Any, *, validate_range: bool = True) -> PackedXtructure:
    """Pack an xtructure_dataclass instance into a compact byte payload + specs."""

    def _pack_obj(obj: Any, cls: Type[Any], prefix: str, out_specs: list[PackedFieldSpec], out_chunks: list[chex.Array]):
        fds: dict[str, FieldDescriptor] = get_field_descriptors(cls)
        for name, desc in fds.items():
            path = f"{prefix}{name}" if prefix else name
            value = getattr(obj, name)
            if is_xtructure_dataclass_type(desc.dtype):
                _pack_obj(value, desc.dtype, prefix=f"{path}.", out_specs=out_specs, out_chunks=out_chunks)
                continue

            arr = jnp.asarray(value)
            shape = tuple(int(x) for x in arr.shape)
            dtype_str = _dtype_to_str(arr.dtype)

            bits = desc.pack_bits
            if bits is not None:
                if validate_range:
                    _validate_bits_range(arr, bits)
                # Normalize bool -> 1 bit.
                if arr.dtype == jnp.bool_:
                    if bits != 1:
                        raise ValueError(f"{path}: bool fields must use pack_bits=1, got {bits}")
                    chunk = to_uint8(arr, 1)
                    out_specs.append(
                        PackedFieldSpec(path=path, kind="bits", dtype=dtype_str, shape=shape, bits=1, nbytes=int(chunk.size))
                    )
                    out_chunks.append(chunk)
                else:
                    # Pack integer values, producing uint8 bytes.
                    chunk = to_uint8(arr, int(bits))
                    out_specs.append(
                        PackedFieldSpec(
                            path=path, kind="bits", dtype=dtype_str, shape=shape, bits=int(bits), nbytes=int(chunk.size)
                        )
                    )
                    out_chunks.append(chunk)
            else:
                chunk = _array_to_raw_bytes(arr)
                out_specs.append(
                    PackedFieldSpec(path=path, kind="raw", dtype=dtype_str, shape=shape, bits=None, nbytes=int(chunk.size))
                )
                out_chunks.append(chunk)

    specs: list[PackedFieldSpec] = []
    chunks: list[chex.Array] = []
    _pack_obj(instance, type(instance), prefix="", out_specs=specs, out_chunks=chunks)

    if not chunks:
        payload = jnp.array([], dtype=jnp.uint8)
    else:
        payload = jnp.concatenate([jnp.asarray(c, dtype=jnp.uint8).reshape(-1) for c in chunks], axis=0)

    return PackedXtructure(
        payload=payload,
        specs=tuple(specs),
        cls_module=type(instance).__module__,
        cls_name=type(instance).__name__,
    )


def unpack_instance(cls: Type[T], packed: PackedXtructure) -> T:
    """Reconstruct an xtructure_dataclass instance from PackedXtructure."""
    payload = jnp.asarray(packed.payload, dtype=jnp.uint8).reshape(-1)
    offset = 0
    leaf_by_path: dict[str, chex.Array] = {}

    for spec in packed.specs:
        seg = payload[offset : offset + spec.nbytes]
        offset += spec.nbytes

        if spec.kind == "raw":
            arr = _raw_bytes_to_array(seg, dtype_str=spec.dtype, shape=spec.shape)
        elif spec.kind == "bits":
            if spec.bits is None:
                raise ValueError(f"{spec.path}: kind='bits' requires bits")
            # from_uint8 returns bool for 1-bit, otherwise uint8.
            unpacked = from_uint8(seg, spec.shape, int(spec.bits))
            dt = jnp.dtype(spec.dtype)
            arr = unpacked.astype(dt) if dt != jnp.bool_ else unpacked.astype(jnp.bool_)
        else:
            raise ValueError(f"Unknown packed spec kind: {spec.kind!r}")

        leaf_by_path[spec.path] = arr

    def _build_obj(target_cls: Type[Any], prefix: str) -> Any:
        fds: dict[str, FieldDescriptor] = get_field_descriptors(target_cls)
        kwargs: dict[str, Any] = {}
        for name, desc in fds.items():
            path = f"{prefix}{name}" if prefix else name
            if is_xtructure_dataclass_type(desc.dtype):
                kwargs[name] = _build_obj(desc.dtype, prefix=f"{path}.")
            else:
                if path not in leaf_by_path:
                    raise KeyError(f"Missing packed field for path '{path}'")
                kwargs[name] = leaf_by_path[path]
        return target_cls(**kwargs)

    return _build_obj(cls, prefix="")

