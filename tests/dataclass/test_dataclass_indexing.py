import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from tests.dataclass.fixtures import NestedData, SimpleData, VectorData
from xtructure.core.xtructure_decorators.indexing import add_indexing_methods


def test_at_set_simple_data():
    original_data = SimpleData.default(shape=(3,))

    data_to_set_scalar = SimpleData(
        id=jnp.array(100, dtype=jnp.uint32), value=jnp.array(99.9, dtype=jnp.float32)
    )

    updated_data_single = original_data.at[1].set(data_to_set_scalar)

    assert updated_data_single.id[0] == original_data.id[0]
    assert updated_data_single.value[0] == original_data.value[0]
    assert updated_data_single.id[1] == data_to_set_scalar.id
    assert updated_data_single.value[1] == data_to_set_scalar.value
    assert updated_data_single.id[2] == original_data.id[2]
    assert updated_data_single.value[2] == original_data.value[2]

    assert original_data.id[1] != data_to_set_scalar.id
    assert original_data.value[1] != data_to_set_scalar.value

    updated_data_scalar_id = original_data.at[0].set(jnp.uint32(50))
    assert updated_data_scalar_id.id[0] == 50
    assert updated_data_scalar_id.value[0] == 50.0
    assert updated_data_scalar_id.id[1] == original_data.id[1]
    assert updated_data_scalar_id.value[1] == original_data.value[1]

    slice_data_to_set = SimpleData.default(shape=(2,))
    updated_data_slice = original_data.at[0:2].set(slice_data_to_set)
    assert updated_data_slice.id[0] == slice_data_to_set.id[0]
    assert updated_data_slice.value[0] == slice_data_to_set.value[0]
    assert updated_data_slice.id[1] == slice_data_to_set.id[1]
    assert updated_data_slice.value[1] == slice_data_to_set.value[1]
    assert updated_data_slice.id[2] == original_data.id[2]
    assert updated_data_slice.value[2] == original_data.value[2]


def test_at_set_vector_data():
    original_data = VectorData.default(shape=(3,))

    vector_to_set = VectorData(
        position=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        velocity=jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32),
    )

    updated_data = original_data.at[1].set(vector_to_set)

    assert jnp.array_equal(updated_data.position[0], original_data.position[0])
    assert jnp.array_equal(updated_data.velocity[0], original_data.velocity[0])

    assert jnp.array_equal(updated_data.position[1], vector_to_set.position)
    assert jnp.array_equal(updated_data.velocity[1], vector_to_set.velocity)

    assert jnp.array_equal(updated_data.position[2], original_data.position[2])
    assert jnp.array_equal(updated_data.velocity[2], original_data.velocity[2])

    assert not jnp.array_equal(original_data.position[1], vector_to_set.position)

    updated_data_scalar = original_data.at[0].set(jnp.float32(7.0))
    assert jnp.all(updated_data_scalar.position[0] == 7.0)
    assert jnp.all(updated_data_scalar.velocity[0] == 7.0)
    assert jnp.array_equal(updated_data_scalar.position[1], original_data.position[1])


def test_at_set_nested_data():
    original_data = NestedData.default(shape=(2,))

    data_to_set_single_nested = NestedData.default()
    data_to_set_single_nested = data_to_set_single_nested.replace(
        simple=SimpleData(
            id=jnp.array(10, dtype=jnp.uint32), value=jnp.array(1.1, dtype=jnp.float32)
        ),
        vector=VectorData(
            position=jnp.ones(3, dtype=jnp.float32), velocity=jnp.ones(3, dtype=jnp.float32) * 2
        ),
    )

    updated_data = original_data.at[0].set(data_to_set_single_nested)

    assert updated_data.simple.id[0] == data_to_set_single_nested.simple.id
    assert updated_data.simple.value[0] == data_to_set_single_nested.simple.value
    assert jnp.array_equal(
        updated_data.vector.position[0], data_to_set_single_nested.vector.position
    )
    assert jnp.array_equal(
        updated_data.vector.velocity[0], data_to_set_single_nested.vector.velocity
    )

    assert updated_data.simple.id[1] == original_data.simple.id[1]
    assert updated_data.simple.value[1] == original_data.simple.value[1]
    assert jnp.array_equal(updated_data.vector.position[1], original_data.vector.position[1])
    assert jnp.array_equal(updated_data.vector.velocity[1], original_data.vector.velocity[1])

    assert original_data.simple.id[0] != data_to_set_single_nested.simple.id


@dataclasses.dataclass
class IndexedData:
    x: jnp.ndarray
    y: jnp.ndarray


IndexedData = add_indexing_methods(IndexedData)


class TestIndexingDecorator(unittest.TestCase):
    def test_set_as_condition_duplicate_indices(self):
        instance = IndexedData(x=jnp.zeros(5), y=jnp.zeros(5))
        indices = jnp.array([0, 0, 1])
        condition = jnp.array([True, False, True])

        updated_instance = instance.at[indices].set_as_condition(condition, 99)

        expected_x = jnp.array([99.0, 99.0, 0.0, 0.0, 0.0])
        self.assertTrue(jnp.array_equal(updated_instance.x, expected_x))
        self.assertTrue(jnp.array_equal(updated_instance.y, expected_x))

    def test_set_as_condition_advanced_indexing_with_duplicates(self):
        instance = IndexedData(x=jnp.zeros((5, 5)), y=jnp.zeros((5, 5)))

        rows = jnp.array([0, 0, 1])
        cols = jnp.array([0, 0, 1])
        indices = (rows, cols)
        condition = jnp.array([True, False, True])

        updated_instance = instance.at[indices].set_as_condition(condition, 88)

        expected_x = jnp.zeros((5, 5))
        expected_x = expected_x.at[0, 0].set(88)
        expected_x = expected_x.at[1, 1].set(88)

        self.assertTrue(jnp.array_equal(updated_instance.x, expected_x))
        self.assertTrue(jnp.array_equal(updated_instance.y, expected_x))

    def test_set_as_condition_with_scalar_value_broadcast(self):
        instance = IndexedData(x=jnp.arange(10), y=jnp.arange(10))

        indices = jnp.array([1, 3, 5, 7])
        condition = jnp.array([True, False, True, False])

        updated_instance = instance.at[indices].set_as_condition(condition, -1)

        expected_x = jnp.array([0, -1, 2, 3, 4, -1, 6, 7, 8, 9])
        self.assertTrue(jnp.array_equal(updated_instance.x, expected_x))
        self.assertTrue(jnp.array_equal(updated_instance.y, expected_x))

    def test_set_as_condition_with_array_values(self):
        instance = IndexedData(x=jnp.zeros(4), y=jnp.zeros(4))

        indices = jnp.array([0, 1, 0, 2])
        condition = jnp.array([True, True, False, True])
        values_to_set = jnp.array([10, 20, 30, 40])

        updated_instance = instance.at[indices].set_as_condition(condition, values_to_set)

        expected_x = jnp.array([10.0, 20.0, 40.0, 0.0])

        self.assertTrue(jnp.array_equal(updated_instance.x, expected_x))
        self.assertTrue(jnp.array_equal(updated_instance.y, expected_x))

    def test_set_as_condition_extreme_randomized(self):
        key = jax.random.PRNGKey(42)
        data_size = 1000
        num_updates = 5000

        instance = IndexedData(
            x=jnp.zeros(data_size, dtype=jnp.float32),
            y=jnp.zeros(data_size, dtype=jnp.float32),
        )

        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        indices = jax.random.randint(subkey1, (num_updates,), 0, data_size)
        condition = jax.random.choice(subkey2, jnp.array([True, False]), (num_updates,))
        values_to_set = jax.random.normal(subkey3, (num_updates,), dtype=jnp.float32)

        expected_x = np.zeros(data_size, dtype=np.float32)
        updates_to_apply = {}
        for i in range(len(indices)):
            idx = indices[i].item()
            if condition[i]:
                if idx not in updates_to_apply:
                    updates_to_apply[idx] = values_to_set[i].item()

        for idx, value in updates_to_apply.items():
            expected_x[idx] = value

        updated_instance = instance.at[indices].set_as_condition(condition, values_to_set)

        self.assertTrue(
            jnp.array_equal(updated_instance.x, jnp.array(expected_x)),
            "Randomized test with array of values failed for field 'x'",
        )
        self.assertTrue(
            jnp.array_equal(updated_instance.y, jnp.array(expected_x)),
            "Randomized test with array of values failed for field 'y'",
        )

    def test_set_as_condition_extreme_randomized_scalar(self):
        key = jax.random.PRNGKey(84)
        data_size = 1000
        num_updates = 5000
        scalar_value = 99.0

        instance = IndexedData(
            x=jnp.zeros(data_size, dtype=jnp.float32),
            y=jnp.zeros(data_size, dtype=jnp.float32),
        )

        key, subkey1, subkey2 = jax.random.split(key, 3)
        indices = jax.random.randint(subkey1, (num_updates,), 0, data_size)
        condition = jax.random.choice(subkey2, jnp.array([True, False]), (num_updates,))

        expected_x = np.zeros(data_size, dtype=np.float32)
        updates_to_apply = {}
        for i in range(len(indices)):
            idx = indices[i].item()
            if condition[i] and idx not in updates_to_apply:
                updates_to_apply[idx] = scalar_value

        for idx, value in updates_to_apply.items():
            expected_x[idx] = value

        updated_instance = instance.at[indices].set_as_condition(condition, scalar_value)

        self.assertTrue(
            jnp.array_equal(updated_instance.x, jnp.array(expected_x)),
            "Randomized test with scalar value failed for field 'x'",
        )
        self.assertTrue(
            jnp.array_equal(updated_instance.y, jnp.array(expected_x)),
            "Randomized test with scalar value failed for field 'y'",
        )
