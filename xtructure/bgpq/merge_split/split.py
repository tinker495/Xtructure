from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def merge_sort_split_kernel(
    ak_ref,
    bk_ref,
    res_key0_ref,
    res_idx0_ref,
    res_key1_ref,
    res_idx1_ref,
):
    """
    Merge and split two sorted arrays while maintaining their relative order.
    This version is Pallas-compliant: writes to output refs, no return.
    Outputs are:
    - res_key0: First N keys of the merged and sorted array.
    - res_idx0: Corresponding original indices for res_key0.
    - res_key1: Remaining keys of the merged and sorted array.
    - res_idx1: Corresponding original indices for res_key1.
    """
    ak_val, bk_val = ak_ref[...], bk_ref[...]
    n_split = ak_val.shape[-1]
    key_concat = jnp.concatenate([ak_val, bk_val])
    indices_payload = jnp.arange(key_concat.shape[0], dtype=jnp.int32)
    sorted_key_full, sorted_idx_full = jax.lax.sort_key_val(key_concat, indices_payload)
    res_key0_ref[...] = sorted_key_full[:n_split]
    res_idx0_ref[...] = sorted_idx_full[:n_split]
    res_key1_ref[...] = sorted_key_full[n_split:]
    res_idx1_ref[...] = sorted_idx_full[n_split:]


@jax.jit
def merge_sort_split_idx(
    ak: jax.Array, bk: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    len_ak_part = ak.shape[-1]
    len_bk_part = bk.shape[-1]
    key_dtype = jnp.result_type(ak.dtype, bk.dtype)
    shape_key0 = jax.ShapeDtypeStruct(shape=ak.shape[:-1] + (len_ak_part,), dtype=key_dtype)
    shape_idx0 = jax.ShapeDtypeStruct(shape=ak.shape[:-1] + (len_ak_part,), dtype=jnp.int32)
    shape_key1 = jax.ShapeDtypeStruct(shape=bk.shape[:-1] + (len_bk_part,), dtype=key_dtype)
    shape_idx1 = jax.ShapeDtypeStruct(shape=bk.shape[:-1] + (len_bk_part,), dtype=jnp.int32)

    return pl.pallas_call(
        merge_sort_split_kernel,
        out_shape=(shape_key0, shape_idx0, shape_key1, shape_idx1),
    )(ak, bk)
