import numpy as np
import pytest

from xtructure.core.layout.bitpack import packed_num_bytes


@pytest.mark.parametrize("active_bits", range(1, 33))
@pytest.mark.parametrize("num_values", [0, 1, 2, 7, 8, 9, 15, 16, 31, 54, 100, 4096])
def test_packed_num_bytes_matches_block_formula(num_values, active_bits):
    """Prevent width-specific specializations from changing packed byte counts.

    Every supported width (1..32 bits) must match ``block_count * bytes_per_block``
    so stored packed shapes remain identical across generic and fast paths.
    """
    block_bits = int(np.lcm(active_bits, 8))
    expected = -(-num_values // (block_bits // active_bits)) * (block_bits // 8)
    assert packed_num_bytes(num_values, active_bits) == expected


@pytest.mark.parametrize(
    "num_values,active_bits,message",
    [
        (-1, 8, "non-negative"),
        (1, 0, "1-32"),
        (1, 33, "1-32"),
    ],
)
def test_packed_num_bytes_rejects_out_of_range(num_values, active_bits, message):
    with pytest.raises(ValueError, match=message):
        packed_num_bytes(num_values, active_bits)
