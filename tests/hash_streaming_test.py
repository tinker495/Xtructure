import os

import chex
import jax
import jax.numpy as jnp

from xtructure import FieldDescriptor, xtructure_dataclass


def test_hash_streaming_matches_raw():
    old_mode = os.environ.get("XTRUCTURE_HASH_STREAM")
    try:
        os.environ["XTRUCTURE_HASH_STREAM"] = "off"

        @xtructure_dataclass(validate=False)
        class HashStreamRaw:
            a: FieldDescriptor.scalar(dtype=jnp.uint8, default=0)
            b: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(3,), fill_value=0)
            c: FieldDescriptor.tensor(dtype=jnp.float32, shape=(2,), fill_value=0.0)

        os.environ["XTRUCTURE_HASH_STREAM"] = "on"

        @xtructure_dataclass(validate=False)
        class HashStreamOn:
            a: FieldDescriptor.scalar(dtype=jnp.uint8, default=0)
            b: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(3,), fill_value=0)
            c: FieldDescriptor.tensor(dtype=jnp.float32, shape=(2,), fill_value=0.0)

    finally:
        if old_mode is None:
            os.environ.pop("XTRUCTURE_HASH_STREAM", None)
        else:
            os.environ["XTRUCTURE_HASH_STREAM"] = old_mode

    key = jax.random.PRNGKey(0)
    k0, k1, k2 = jax.random.split(key, 3)
    a = jax.random.randint(k0, (), 0, 256, dtype=jnp.uint8)
    b = jax.random.randint(k1, (3,), 0, 2**16, dtype=jnp.uint16)
    c = jax.random.normal(k2, (2,), dtype=jnp.float32)

    raw = HashStreamRaw(a=a, b=b, c=c)
    streamed = HashStreamOn(a=a, b=b, c=c)

    chex.assert_trees_all_equal(raw.uint32ed, streamed.uint32ed)

    h_raw = raw.hash(0)
    h_stream = streamed.hash(0)
    chex.assert_trees_all_equal(h_raw, h_stream)

    hp_raw = raw.hash_pair(123)
    hp_stream = streamed.hash_pair(123)
    chex.assert_trees_all_equal(hp_raw, hp_stream)
