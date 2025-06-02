import pytest
import torch
from quanproto.metrics.contrastivity import (
    activation_entropy,
    get_feature_vectors,
    inter_class_vector_distance,
    intra_bb_location_change,
    intra_mask_activation_change,
    intra_mask_location_change,
    intra_vector_distance,
    max_activation_location_change,
)


def test_intra_mask_location_change_basic():
    # Create a batch of maps with 2 x 3 x 4 x 4
    # fmt: off
    map_batch = torch.tensor(
        [
            [
                [[1, 1, 0, 0],
                 [1, 1, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],

                [[0, 0, 1, 1],
                 [0, 0, 1, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],

            ],
            [
                [[1, 0, 0, 1],
                 [0, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 1, 1, 1]],

                [[0, 0, 0, 1],
                 [0, 0, 0, 0],
                 [1, 0, 0, 0],
                 [1, 1, 1, 1]],
            ],
        ]
    )
    # fmt: on

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 2, 4, 4)

    output = intra_mask_location_change(map_batch)

    # Check the output
    expected_output = torch.tensor([1.0, 0.5])

    assert torch.allclose(output, expected_output)


def test_intra_bb_location_change_basic():

    bbox_batch = torch.tensor(
        [
            [[0, 0, 5, 4], [1, 2, 5, 6]],
        ]
    )

    output = intra_bb_location_change(bbox_batch)

    expected_output = torch.tensor([0.71428])

    assert torch.allclose(output, expected_output)


def test_intra_mask_activation_change_basic():
    # Create a batch of maps with 2 x 3 x 4 x 4
    # fmt: off
    map_batch = torch.tensor(
        [
            [
                [[2, 2, 0, 0],
                 [2, 2, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],

                [[0, 0, 1, 1],
                 [0, 0, 1, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],

            ],
            [
                [[1, 0, 0, 1],
                 [0, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 1, 1, 1]],

                [[0, 0, 0, 5],
                 [0, 0, 0, 0],
                 [5, 0, 0, 0],
                 [5, 5, 5, 5]],
            ],
        ]
    )
    # fmt: on

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 2, 4, 4)

    output = intra_mask_activation_change(map_batch)

    # Check the output
    expected_output = torch.tensor([0.5, 0.8])

    assert torch.allclose(output, expected_output)


def test_max_activation_location_change_basic():
    # Create a batch of maps with 2 x 3 x 4 x 4
    # fmt: off
    map_batch = torch.tensor(
        [
            [
                [[2, 2, 0, 0],
                 [2, 2, 0, 0],
                 [0, 0, 5, 0],
                 [0, 0, 0, 0]],

                [[5, 0, 1, 1],
                 [0, 0, 1, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],

            ],
            [
                [[1, 0, 0, 2],
                 [0, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 1, 1, 1]],

                [[0, 0, 0, 5],
                 [0, 0, 0, 0],
                 [5, 0, 0, 0],
                 [10, 5, 5, 5]],
            ],
        ]
    )
    # fmt: on

    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 2, 4, 4)

    output = max_activation_location_change(map_batch)

    # Check the output
    expected_output = torch.tensor([4.0, 6.0])

    assert torch.allclose(output, expected_output)


def test_get_feature_vectors():
    # fmt: off
    map_batch = torch.tensor(
        [
            [
                [[2, 2, 0, 0],
                 [2, 2, 0, 0],
                 [0, 0, 5, 0],
                 [0, 0, 0, 0]],

                [[5, 0, 1, 1],
                 [0, 0, 1, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],

            ],
            [
                [[1, 0, 0, 2],
                 [0, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 1, 1, 1]],

                [[0, 0, 0, 5],
                 [0, 0, 0, 0],
                 [5, 0, 0, 0],
                 [10, 5, 5, 5]],
            ],
        ]
    )
    # check the dimensions of the map_batch
    assert map_batch.shape == (2, 2, 4, 4)
    
    feature_vectors = torch.tensor(
        [
            [
                [[2, 2, 0, 0],
                 [2, 2, 0, 0],
                 [0, 0, 5, 0],
                 [0, 0, 0, 0]],

                [[5, 0, 1, 1],
                 [0, 0, 1, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
                
                [[5, 0, 1, 1],
                 [1, 0, 1, 1],
                 [0, 0, 1, 0],
                 [1, 0, 0, 0]],

            ],
            [
                [[1, 0, 0, 2],
                 [0, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 1, 1, 1]],

                [[0, 0, 0, 5],
                 [0, 0, 0, 0],
                 [5, 0, 0, 0],
                 [10, 5, 5, 5]],

                [[5, 0, 1, 1],
                 [0, 0, 1, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
            ],
        ]
    )

    assert feature_vectors.shape == (2, 3, 4, 4)

    # fmt: on

    output = get_feature_vectors(map_batch, feature_vectors)

    # Check the output 2 x 2 x 3
    expected_output = torch.tensor([[[5, 0, 1], [2, 5, 5]], [[2, 5, 1], [0, 10, 0]]])

    assert torch.allclose(output, expected_output)


def test_intra_vector_distance():
    # fmt: off
    feature_vectors = torch.tensor(
        [[
            [1, 1, 1], 
            [1, 1, 1]],

            [[0, 0, 1],
             [1, 0, 0]],

            [[-1, 1, 1],
             [1, -1, -1]],
        ]
    ).float()
    # fmt: on
    output = intra_vector_distance(feature_vectors)

    # Check the output
    expected_output = torch.tensor([0, 1, 2]).float()

    assert torch.allclose(output, expected_output)


def test_inter_class_vector_distance():
    # fmt: off
    feature_vectors = torch.tensor(
        [[
            [1, 1, 1], 
            [1, 1, 1], 
            [1, 1, 1]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ]
    ).float()

    class_identity = torch.tensor([[1, 0], [0, 1]])
    # fmt: on

    output = inter_class_vector_distance(feature_vectors, class_identity)

    # Check the output 2 x 2 x 3
    expected_output = torch.tensor([1, 1]).float()

    assert torch.allclose(output, expected_output)


def test_activation_entropy():
    # fmt: off
    # Create a batch of maps with 2 x 5
    similarities = torch.tensor(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 1.0],
        ]
    )
    # fmt: on

    # check dim
    assert similarities.shape == (2, 5)

    output = activation_entropy(similarities)

    # Check the output
    expected_output = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])

    assert torch.allclose(output, expected_output)


if __name__ == "__main__":
    pytest.main()
