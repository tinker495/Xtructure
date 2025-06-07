import time
from functools import partial

import jax.numpy as jnp
import jax.random as jr
from jax import jit

from .merge_split import merge_arrays_indices_loop, merge_arrays_parallel


def run_correctness_tests():
    """Runs a series of correctness tests with small, fixed inputs."""
    print("\n--- Running correctness tests ---")
    # Test case 1: Identical arrays
    a1 = jnp.array([1, 2, 3, 4])
    b1 = jnp.array([1, 2, 3, 4])
    merged_keys1, merged_indices1 = merge_arrays_indices_loop(a1, b1)
    concatenated1 = jnp.concatenate([a1, b1])
    assert jnp.array_equal(merged_keys1, concatenated1[merged_indices1])
    print("✅ Test 1 (Identical) PASSED")

    # Test case 2: Interleaved arrays
    a2 = jnp.array([1, 5, 9])
    b2 = jnp.array([2, 6, 10])
    merged_keys2, merged_indices2 = merge_arrays_indices_loop(a2, b2)
    concatenated2 = jnp.concatenate([a2, b2])
    assert jnp.array_equal(merged_keys2, concatenated2[merged_indices2])
    print("✅ Test 2 (Interleaved) PASSED")

    # Test case 3: One array exhausted first
    a3 = jnp.array([1, 2])
    b3 = jnp.array([3, 4, 5, 6])
    merged_keys3, merged_indices3 = merge_arrays_indices_loop(a3, b3)
    concatenated3 = jnp.concatenate([a3, b3])
    assert jnp.array_equal(merged_keys3, concatenated3[merged_indices3])
    print("✅ Test 3 (Exhaustion) PASSED")

    # Test case 4: Empty array
    a4 = jnp.array([], dtype=jnp.int32)
    b4 = jnp.array([1, 2, 3])
    merged_keys4a, merged_indices4a = merge_arrays_indices_loop(a4, b4)
    concatenated4a = jnp.concatenate([a4, b4])
    assert jnp.array_equal(merged_keys4a, concatenated4a[merged_indices4a])
    print("✅ Test 4a (Empty Left) PASSED")

    merged_keys4b, merged_indices4b = merge_arrays_indices_loop(b4, a4)
    concatenated4b = jnp.concatenate([b4, a4])
    assert jnp.array_equal(merged_keys4b, concatenated4b[merged_indices4b])
    print("✅ Test 4b (Empty Right) PASSED")

    # Test case 5: Arrays with duplicate values across them
    a5 = jnp.array([10, 20, 30])
    b5 = jnp.array([10, 25, 30])
    merged_keys5, merged_indices5 = merge_arrays_indices_loop(a5, b5)
    concatenated5 = jnp.concatenate([a5, b5])
    assert jnp.array_equal(merged_keys5, concatenated5[merged_indices5])
    print("✅ Test 5 (Duplicates) PASSED")
    print("--- All correctness tests passed ---")


@partial(jit, static_argnums=(1, 2, 3))
def generate_sorted_test_data(key, size_ak, size_bk, dtype):
    """JIT-compiled function to generate and sort random test arrays."""
    key_ak, key_bk = jr.split(key)

    if jnp.issubdtype(dtype, jnp.integer):
        ak_rand = jr.randint(key_ak, (size_ak,), minval=0, maxval=max(1, size_ak * 10), dtype=dtype)
        bk_rand = jr.randint(key_bk, (size_bk,), minval=0, maxval=max(1, size_bk * 10), dtype=dtype)
    elif jnp.issubdtype(dtype, jnp.floating):
        ak_rand = jr.uniform(
            key_ak, (size_ak,), dtype=dtype, minval=0.0, maxval=float(max(1, size_ak * 10))
        )
        bk_rand = jr.uniform(
            key_bk, (size_bk,), dtype=dtype, minval=0.0, maxval=float(max(1, size_bk * 10))
        )
    else:
        raise TypeError(f"Unsupported dtype for random generation: {dtype}")

    ak_sorted = jnp.sort(ak_rand)
    bk_sorted = jnp.sort(bk_rand)
    return ak_sorted, bk_sorted


@jit
def jax_baseline_merge(ak, bk):
    """A JIT-compiled baseline merge implementation using standard JAX ops."""
    return jnp.sort(jnp.concatenate([ak, bk]))


def verify_and_time_merge(key, size_ak, size_bk, dtype=jnp.int32):
    """Generates random data and benchmarks all merge implementations."""
    print(f"\nTesting with ak_size={size_ak}, bk_size={size_bk}, dtype={dtype}")

    # Use the JIT-compiled function for faster data generation
    ak_sorted, bk_sorted = generate_sorted_test_data(key, size_ak, size_bk, dtype)
    ak_sorted.block_until_ready()
    bk_sorted.block_until_ready()

    if size_ak < 10 and size_bk < 10:
        print(f"  Sorted ak: {ak_sorted}")
        print(f"  Sorted bk: {bk_sorted}")

    implementations_to_test = {
        "pallas_loop": merge_arrays_indices_loop,
        "pallas_parallel": merge_arrays_parallel,
    }

    reference_merged_jax = jnp.sort(jnp.concatenate([ak_sorted, bk_sorted]))
    reference_merged_jax.block_until_ready()
    concatenated_inputs = jnp.concatenate([ak_sorted, bk_sorted])

    all_passed = True
    for name, merge_fn in implementations_to_test.items():
        print(f"  Verifying implementation: {name}")
        try:
            merged_keys, merged_indices = merge_fn(ak_sorted, bk_sorted)
            merged_keys.block_until_ready()
            merged_indices.block_until_ready()
            reconstructed = concatenated_inputs[merged_indices]
            reconstructed.block_until_ready()
            assert jnp.array_equal(merged_keys, reference_merged_jax)
            assert jnp.array_equal(reconstructed, reference_merged_jax)
            print("    ✅ Correctness check PASSED.")
        except AssertionError as e:
            all_passed = False
            print(f"    ❌ Correctness check FAILED for {name}.")
            if size_ak < 20 and size_bk < 20:
                print(f"      Pallas merged keys:   {merged_keys}")
                print(f"      Pallas reconstructed: {reconstructed}")
                print(f"      JAX reference sorted: {reference_merged_jax}")
                print(f"      Pallas indices:       {merged_indices}")
            print(f"    ❌ {e}")
        except Exception as e:
            all_passed = False
            print(f"    ❌ An exception occurred during {name} execution: {e}")

    if not all_passed:
        print("  Skipping timing due to correctness failure.")
        return

    print("\n  --- Timing Comparison ---")

    # JAX baseline
    _ = jax_baseline_merge(ak_sorted, bk_sorted).block_until_ready()
    start_time_jax = time.perf_counter()
    for _ in range(10):
        jax_output_timing = jax_baseline_merge(ak_sorted, bk_sorted)
    jax_output_timing.block_until_ready()
    end_time_jax = time.perf_counter()
    jax_time = (end_time_jax - start_time_jax) / 10
    print(f"  ⏱️ JAX Baseline (JIT) avg time:             {jax_time*1000:.4f} ms")

    # Pallas versions
    for name, merge_fn in implementations_to_test.items():
        _keys, _indices = merge_fn(ak_sorted, bk_sorted)
        _keys.block_until_ready()
        _indices.block_until_ready()
        start_time_pallas = time.perf_counter()
        for _ in range(10):
            pallas_keys_timing, pallas_indices_timing = merge_fn(ak_sorted, bk_sorted)
        pallas_keys_timing.block_until_ready()
        pallas_indices_timing.block_until_ready()
        end_time_pallas = time.perf_counter()
        pallas_time = (end_time_pallas - start_time_pallas) / 10
        print(f"  ⏱️ Pallas '{name}' avg time: {pallas_time*1000:.4f} ms")


if __name__ == "__main__":
    run_correctness_tests()

    print("\n\n--- Running benchmarks with random values and timing ---")
    master_key = jr.PRNGKey(42)

    sizes_to_test = [8, 200, 1000, 5000, int(1e5)]
    dtypes_to_test = [jnp.int32, jnp.float32]

    for size in sizes_to_test:
        for dtype in dtypes_to_test:
            key, subkey = jr.split(master_key)
            verify_and_time_merge(subkey, size_ak=size, size_bk=size, dtype=dtype)

    print("\n--- Benchmark complete ---")
