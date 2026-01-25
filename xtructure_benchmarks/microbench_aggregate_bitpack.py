import argparse
import os
import time

import jax
import jax.numpy as jnp

from xtructure import FieldDescriptor, xtructure_dataclass


@xtructure_dataclass(validate=False, aggregate_bitpack=True)
class AggBitpackBench:
    # ~928 bytes packed payload (<= 1e3 target)
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(1024,), bits=1, fill_value=False)
    faces: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(256, 3), bits=3, fill_value=0)
    codes: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(256,), bits=12, fill_value=0)
    ids: FieldDescriptor.tensor(dtype=jnp.uint32, shape=(32,), bits=32, fill_value=0)


def _time_call(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    t1 = time.perf_counter()
    return t1 - t0


def main():
    parser = argparse.ArgumentParser(description="Microbench aggregate_bitpack")
    parser.add_argument("--backend", choices=["auto", "pallas", "xla"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--word-tile", type=int, default=4)
    args = parser.parse_args()

    os.environ["XTRUCTURE_AGG_BITPACK_PACK_BACKEND"] = args.backend
    os.environ["XTRUCTURE_AGG_BITPACK_PACK_WORD_TILE"] = str(args.word_tile)

    print("jax backend:", jax.default_backend())
    print("devices:", jax.devices())
    print("pack backend env:", os.environ["XTRUCTURE_AGG_BITPACK_PACK_BACKEND"])
    print("word_tile env:", os.environ["XTRUCTURE_AGG_BITPACK_PACK_WORD_TILE"])

    batch = int(args.batch_size)
    key = jax.random.PRNGKey(0)
    k0, k1, k2, k3 = jax.random.split(key, 4)

    x = AggBitpackBench.default(shape=(batch,))
    x = x.replace(
        flags=jax.random.bernoulli(k0, p=0.5, shape=(batch, 1024)),
        faces=jax.random.randint(k1, (batch, 256, 3), 0, 8, dtype=jnp.uint8),
        codes=jax.random.randint(k2, (batch, 256), 0, 4096, dtype=jnp.uint16),
        ids=jax.random.bits(k3, (batch, 32), dtype=jnp.uint32),
    )

    pack_words = jax.jit(lambda v: v.packed.words)
    t_first = _time_call(pack_words, x)
    t_second = _time_call(pack_words, x)
    print(f"pack.words first-call (compile+run): {t_first:.3f}s")
    print(f"pack.words second-call (run): {t_second:.6f}s")

    # Unpack a hot-ish field.
    p = x.packed
    unpack_codes = jax.jit(lambda packed: packed.unpack_field("codes"))
    t_u_first = _time_call(unpack_codes, p)
    t_u_second = _time_call(unpack_codes, p)
    print(f"unpack_field('codes') first-call (compile+run): {t_u_first:.3f}s")
    print(f"unpack_field('codes') second-call (run): {t_u_second:.6f}s")


if __name__ == "__main__":
    main()
