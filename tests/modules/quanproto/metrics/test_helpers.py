import pytest
import torch
from quanproto.metrics.helpers import (
    binary_mask,
    bounding_box,
    min_max_norm_mask,
    percentile_mask,
)


def test_percentile_mask_basic():
    # Create a map with 3 x 3
    map = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    ).float()

    output = percentile_mask(map, percentile=50.0)

    # Check the output
    expected_output = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 1, 1],
        ]
    ).float()

    assert torch.allclose(output, expected_output)


def test_binary_mask_basic():
    # Create a map with 3 x 3
    map = torch.tensor(
        [
            [0.1, 0, -1],
            [0, 8, 0],
            [-10, 0, 5],
        ]
    ).float()

    output = binary_mask(map)

    # Check the output
    expected_output = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    ).float()

    assert torch.allclose(output, expected_output)


def test_min_max_norm_mask_basic():
    # Create a map with 3 x 3
    map = torch.tensor(
        [
            [5, 5, 5],
            [5, 5, 6],
            [7, 8, 9],
        ]
    ).float()

    output = min_max_norm_mask(map, threshold=0.5)

    # Check the output
    expected_output = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 1],
        ]
    ).float()

    assert torch.allclose(output, expected_output)


def test_bounding_box_basic():

    map = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 5, 6, 6, 6, 0, 0],
            [0, 8, 6, 6, 6, 0, 0],
            [0, 5, 6, 6, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    ).float()

    output = bounding_box(map, mask_fn=binary_mask)

    # Check the output min_x, min_y, max_x, max_y
    expected_output = (1, 1, 4, 3)

    assert output == expected_output


if __name__ == "__main__":
    pytest.main()
