import pytest
import torch
from quanproto.metrics.complexity import (
    background_overlap,
    boundingbox_consistency,
    inside_outside_relevance,
    map_consistency,
    mask_intersection_over_union,
    mask_overlap,
    outside_inside_relevance_ratio,
)


def test_mask_intersection_over_union_basic():
    # Test a case with 33% IOU

    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
            ],
            [
                [[1, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1]],
                [[0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]],
                [[0, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 1, 0, 1]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        ]
    )

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = mask_intersection_over_union(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor(
        [[0.333333, 0.333333, 0.333333], [0.333333, 0.333333, 0.333333]]
    )

    assert torch.allclose(output, expected_output)


def test_mask_intersection_over_union_all_zero():
    # Test a case with no overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        ]
    )

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = mask_intersection_over_union(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    assert torch.allclose(output, expected_output)


def test_mask_intersection_over_union_no_overlap():
    # Test a case with no overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]],
                [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
                [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
            ],
            [
                [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]],
                [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        ]
    )

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = mask_intersection_over_union(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    assert torch.allclose(output, expected_output)


def test_mask_overlap_basic():
    # Test a case with 50% overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
            ],
            [
                [[1, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1]],
                [[0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]],
                [[0, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 1, 0, 1]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        ]
    )

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = mask_overlap(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

    assert torch.allclose(output, expected_output)


def test_mask_overlap_all_zero():
    # Test a case with no overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        ]
    )

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = mask_overlap(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    assert torch.allclose(output, expected_output)


def test_mask_overlap_no_overlap():
    # Test a case with no overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]],
                [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
                [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
            ],
            [
                [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]],
                [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        ]
    )

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = mask_overlap(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    assert torch.allclose(output, expected_output)


def test_background_overlap_basic():
    # Test a case with 50% overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
            ],
            [
                [[1, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1]],
                [[0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]],
                [[0, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 1, 0, 1]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        ]
    )

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = background_overlap(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

    assert torch.allclose(output, expected_output)


def test_background_overlap_all_zero():
    # Test a case with no overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        ]
    )

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = background_overlap(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    assert torch.allclose(output, expected_output)


def test_background_overlap_no_overlap():
    # Test a case with no overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]],
                [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
                [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
            ],
            [
                [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]],
                [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        ]
    )

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = background_overlap(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    assert torch.allclose(output, expected_output)


def test_outside_inside_relevance_ratio_basic():
    # Test a case with 50% overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    # fmt: off
    map_batch = torch.tensor(
        [
            [
                [[5, 1, 0, 0], [1, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 5, 2, 0], [0, 2, 5, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 9, 1], [0, 0, 3, 1]],
            ],
            [
                [[5, 0, 0, 5],
                 [0, 0, 0, 0],
                 [0, 5, 0, 0],
                 [0, 5, 5, 5]],

                [[0, 0, 0, 1],
                 [0, 0, 0, 0],
                 [4, 0, 0, 0],
                 [1, 4, 4, 1]],

                [[1, 2, 2, 1],
                 [2, 1, 2, 2],
                 [2, 2, 1, 2],
                 [1, 2, 2, 1]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],

            [[1, 0, 0, 1],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [1, 0, 0, 1]],
        ]
    )
    # fmt: on

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = outside_inside_relevance_ratio(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor([[1 / 5, 2 / 5, 2 / 5], [1, 4, 2]])

    assert torch.allclose(output, expected_output)


def test_outside_inside_relevance_ratio_all_zero():
    # Test a case with no overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        ]
    )

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = outside_inside_relevance_ratio(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor(
        [
            [float("inf"), float("inf"), float("inf")],
            [float("inf"), float("inf"), float("inf")],
        ]
    )

    assert torch.allclose(output, expected_output)


def test_outside_inside_relevance_ratio_no_overlap():
    # Test a case with no overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]],
                [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
                [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
            ],
            [
                [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]],
                [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        ]
    )

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = outside_inside_relevance_ratio(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor(
        [
            [float("inf"), float("inf"), float("inf")],
            [float("inf"), float("inf"), float("inf")],
        ]
    )

    assert torch.allclose(output, expected_output)


def test_outside_inside_relevance_ratio_all_inside():
    # Test a case with no overlap

    # fmt: off
    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]],

                [[2, 0, 0, 0],
                 [0, 2, 0, 0],
                 [0, 0, 2, 0],
                 [0, 0, 0, 2]],

                [[1, 0, 0, 0],
                 [0, 2, 0, 0],
                 [0, 0, 3, 0],
                 [0, 0, 0, 4]],
            ],
            [
                [[1, 0, 0, 2],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [2, 0, 0, 1]],

                [[2, 0, 0, 3],
                 [0, 3, 0, 0],
                 [0, 0, 3, 0],
                 [2, 0, 0, 3]],

                [[10, 0, 0, 9],
                 [0, 8, 0, 0],
                 [0, 0, 8, 0],
                 [6, 0, 0, 9]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],

            [[1, 0, 0, 1],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [1, 0, 0, 1]],
        ]
    )
    # fmt: on

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = outside_inside_relevance_ratio(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    assert torch.allclose(output, expected_output)


def test_inside_outside_relevance_basic():
    # Test a case with 50% overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    # fmt: off
    map_batch = torch.tensor(
        [
            [
                [[5, 1, 0, 0],
                 [1, 5, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],

                [[0, 0, 0, 0],
                 [0, 5, 2, 0],
                 [0, 2, 5, 0],
                 [0, 0, 0, 0]],

                [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 9, 1],
                 [0, 0, 3, 1]],
            ],
            [
                [[5, 0, 0, 5],
                 [0, 0, 0, 0],
                 [0, 5, 0, 0],
                 [0, 5, 5, 5]],

                [[0, 0, 0, 1],
                 [0, 0, 0, 0],
                 [4, 0, 0, 0],
                 [1, 4, 4, 1]],

                [[1, 2, 2, 1],
                 [2, 1, 2, 2],
                 [2, 2, 1, 2],
                 [1, 2, 2, 1]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],

            [[1, 0, 0, 1],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [1, 0, 0, 1]],
        ]
    )
    # fmt: on

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = inside_outside_relevance(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor([[0.8, 0.6, 0.333333], [0, -0.75, -0.5]])

    assert torch.allclose(output, expected_output)


def test_inside_outside_relevance_all_zero():
    # Test a case with no overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            [
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        ]
    )

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = inside_outside_relevance(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    assert torch.allclose(output, expected_output)


def test_inside_outside_relevance_no_overlap():
    # Test a case with no overlap

    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]],
                [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
                [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
            ],
            [
                [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]],
                [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
        ]
    )

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = inside_outside_relevance(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ]
    )

    assert torch.allclose(output, expected_output)


def test_inside_outside_relevance_all_inside():
    # Test a case with no overlap

    # fmt: off
    # Create a batch of maps with 2 x 3 x 4 x 4
    map_batch = torch.tensor(
        [
            [
                [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]],

                [[2, 0, 0, 0],
                 [0, 2, 0, 0],
                 [0, 0, 2, 0],
                 [0, 0, 0, 2]],

                [[2, 0, 0, 0],
                 [0, 2, 0, 0],
                 [0, 0, 4, 0],
                 [0, 0, 0, 4]],
            ],
            [
                [[2, 0, 0, 2],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [2, 0, 0, 1]],

                [[3, 0, 0, 6],
                 [0, 3, 0, 0],
                 [0, 0, 3, 0],
                 [6, 0, 0, 6]],

                [[8, 0, 0, 2],
                 [0, 8, 0, 0],
                 [0, 0, 8, 0],
                 [8, 0, 0, 2]],
            ],
        ]
    )

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 3, 4, 4)

    # Create a batch of masks with 2 x 4 x 4
    mask_batch = torch.tensor(
        [
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],

            [[1, 0, 0, 1],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [1, 0, 0, 1]],
        ]
    )
    # fmt: on

    # check the dimensions of the mask_batch
    assert mask_batch.shape == (2, 4, 4)

    # Compute the intersection over union
    output = inside_outside_relevance(map_batch, mask_batch)

    # Check the output
    expected_output = torch.tensor(
        [
            [1.0, 1.0, 0.75],
            [0.75, 0.75, 0.75],
        ]
    )

    assert torch.allclose(output, expected_output)


def test_boundingbox_consistency_basic():
    # bounding box batch (B x N x 4)
    # fmt: off
    # min_x, min_y, max_x, max_y
    bbox_batch = torch.tensor(
        [
            [[0, 0, 5, 4],
             [1, 2, 5, 6]],
        ]
    )

    assert bbox_batch.shape == (1, 2, 4)

    # part locations batch (B x K x 2)
    partlocs = torch.tensor(
        [
            [[3, 1],
             [3, 3],
             [0, 3], # on the edge of bb1 
             [6, 3],
             [3,5]],
        ]
    )

    assert partlocs.shape == (1, 5, 2)

    # partloc ids (B x K)
    part_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
        ]
    )

    assert part_ids.shape == (1, 5)

    output = boundingbox_consistency(bbox_batch, partlocs, part_ids)

    # B x N x K
    expected_output = torch.tensor(
        [
            [[1, 2, 3, 0, 0],
             [0, 2, 0, 0, 5]],
        ]
    )
    assert expected_output.shape == (1, 2, 5)

    assert torch.allclose(output, expected_output)
    # fmt: on


def test_map_consistency_basic():
    # fmt: off
    # map batch (1 x 2 x 7 x 7)
    map_batch = torch.tensor(
        [[
            [[1,1,1,1,1,1,0],
             [1,1,1,1,1,1,0],
             [1,1,1,1,1,1,0],
             [1,1,1,1,1,1,0],
             [1,1,1,1,1,1,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0]],

            [[0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,1,1,1,1,1,0],
             [0,1,1,1,1,1,0],
             [0,1,1,1,1,1,0],
             [0,1,1,1,1,1,0],
             [0,1,1,1,1,1,0]],
        ]]
    )

    assert map_batch.shape == (1, 2, 7, 7)

    # part locations batch (B x K x 2)
    partlocs = torch.tensor(
        [
            [[3, 1],
             [3, 3],
             [0, 3], # on the edge of bb1 
             [6, 3],
             [3,5]],
        ]
    )

    assert partlocs.shape == (1, 5, 2)

    # partloc ids (B x K)
    part_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
        ]
    )

    assert part_ids.shape == (1, 5)

    output = map_consistency(map_batch, partlocs, part_ids)

    # B x N x K
    expected_output = torch.tensor(
        [
            [[1, 2, 3, 0, 0],
             [0, 2, 0, 0, 5]],
        ]
    )
    assert expected_output.shape == (1, 2, 5)

    assert torch.allclose(output, expected_output)
    # fmt: on


if __name__ == "__main__":
    pytest.main()
