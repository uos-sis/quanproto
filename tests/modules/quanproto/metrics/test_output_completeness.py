import pytest
import torch
from quanproto.metrics.output_completeness import (
    activation_change,
    add_gaussian_noise,
    bb_location_change,
    gaussian_noise_mask,
    mask_location_change,
    max_activation_change,
    max_activation_location_change,
    rank_change,
)


def test_add_gaussian_noise():
    # Create an image with 3 x 3
    image = torch.tensor(
        [
            [255, 255, 0],
            [255, 255, 0],
            [0, 0, 0],
        ]
    ).float()
    # lower_x, lower_y, upper_x, upper_y
    bb = (0, 0, 1, 1)
    mean = 0.0
    std = 0.5

    output = add_gaussian_noise(image, bb, mean, std)

    # Check if the output has changed in the bounding box
    assert not torch.allclose(output[2, 0:2], image[2, 0:2])
    assert not torch.allclose(output[0:2, 2], image[0:2, 2])

    # Check if the output has changed outside the bounding box
    assert torch.allclose(output[0:2, 0:2], image[0:2, 0:2])


def test_gaussian_noise_mask():
    # Create an image with 3 x 3
    image = torch.tensor(
        [
            [255, 255, 0],
            [255, 255, 0],
            [0, 0, 0],
        ]
    ).float()
    # lower_x, lower_y, upper_x, upper_y
    bb = (0, 0, 1, 1)
    mean = 0.0
    std = 0.5

    output = gaussian_noise_mask(image, bb, mean, std)

    expected_bb_otuput = torch.tensor(
        [
            [0, 0],
            [0, 0],
        ]
    ).float()

    # Check if the output has changed in the bounding box
    assert not torch.allclose(output[2, 0:2], image[2, 0:2])
    assert not torch.allclose(output[0:2, 2], image[0:2, 2])

    # Check if the output has changed outside the bounding box
    assert torch.allclose(output[0:2, 0:2], expected_bb_otuput)


def test_bb_location_change():
    # bounding box batch (B x N x 4)
    # fmt: off
    # min_x, min_y, max_x, max_y
    bbox_batch1 = torch.tensor(
        [
            [[0, 0, 5, 4],
             [1, 2, 5, 4],
             [1, 2, 5, 6]],
        ]
    )
    bbox_batch2 = torch.tensor(
        [
            [[0, 0, 5, 4],
             [1, 4, 5, 6],
             [1, 4, 5, 6]],
        ]
    )
    # fmt: on

    output = bb_location_change(bbox_batch1, bbox_batch2)

    # Check the output
    expected_output = torch.tensor([0.0, 1.0, 0.5])

    assert torch.allclose(output, expected_output)


def test_activation_change_basic():
    # activation map batch (B x N x H x W) (1 x 2 x 4 x 4)
    # fmt: off
    map_batch1 = torch.tensor(
        [
            [[2, 2, 0, 0],
             [2, 2, 0, 0],
             [0, 0, 5, 0],
             [0, 0, 0, 0]],

            [[5, 0, 1, 1],
             [0, 0, 1, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0]], 
        ]).float()
    map_batch1 = map_batch1.unsqueeze(0)

    map_batch2 = torch.tensor(
        [
            [[1, 1, 0, 0],
             [1, 1, 0, 0],
             [0, 0, 2.5, 0],
             [0, 0, 0, 0]],

            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]], 
        ]).float()
    map_batch2 = map_batch2.unsqueeze(0)
    # fmt: on
    assert map_batch1.shape == (1, 2, 4, 4)

    output = activation_change(map_batch1, map_batch2)

    # Check the output
    expected_output = torch.tensor([0.5, 1.0])

    assert torch.allclose(output, expected_output)


def test_max_activation_location_change_basic():
    # fmt: off
    map_batch1= torch.tensor(
            [
                [[2, 2, 0, 0],
                 [2, 2, 0, 0],
                 [0, 0, 5, 0],
                 [0, 0, 0, 0]],

                [[1, 0, 0, 2],
                 [0, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 1, 1, 1]],
            ],
    )
    map_batch1= map_batch1.unsqueeze(0)

    map_batch2 = torch.tensor(
            [
                [[5, 0, 1, 1],
                 [0, 0, 1, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],

                [[0, 0, 0, 5],
                 [0, 0, 0, 0],
                 [5, 0, 0, 0],
                 [10, 5, 5, 5]],
            ]
        
    )
    map_batch2 = map_batch2.unsqueeze(0)
    # fmt: on

    assert map_batch1.shape == (1, 2, 4, 4)
    assert map_batch2.shape == (1, 2, 4, 4)

    output = max_activation_location_change(map_batch1, map_batch2)

    expected_output = torch.tensor([4, 6])

    assert torch.allclose(output, expected_output)


def test_max_activation_location_change_basic():
    # fmt: off
    map_batch1= torch.tensor(
            [
                [[2, 2, 0, 0],
                 [2, 2, 0, 0],
                 [0, 0, 5, 0],
                 [0, 0, 0, 0]],

                [[1, 0, 0, 2.5],
                 [0, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 1, 1, 1]],
            ],
    )
    map_batch1= map_batch1.unsqueeze(0)

    map_batch2 = torch.tensor(
            [
                [[2.5, 0, 1, 1],
                 [0, 0, 1, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],

                [[0, 0, 0, 5],
                 [0, 0, 0, 0],
                 [5, 0, 0, 0],
                 [10, 5, 5, 5]],
            ]
        
    )
    map_batch2 = map_batch2.unsqueeze(0)
    # fmt: on

    assert map_batch1.shape == (1, 2, 4, 4)
    assert map_batch2.shape == (1, 2, 4, 4)

    output = max_activation_change(map_batch1, map_batch2).squeeze()

    expected_output = torch.tensor([0.5, 3.0])

    assert torch.allclose(output, expected_output)


def test_rank_change_basic():
    # fmt: off

    # B x K = 1 x 3
    topk_indices = torch.tensor(
        [
            [0, 1, 2],
        ]
    )

    # B x K x N = 1 x 3 x 4
    new_scores = torch.tensor(
        [
            [1.0, 0.2, 0.1, 0.3],   
            [0.2, 0.4, 0.5, 0.3],
            [0.1, 0.2, 1.0, 0.3],
        ]
    )

    output = rank_change(topk_indices, new_scores)

    # Check the output
    expected_output = torch.tensor([[0, 1, 1],[3, 0, 2],[3,1,2]])

    assert torch.allclose(output, expected_output)


def test_mask_location_change_basic():
    # Create a batch of maps with 2 x 3 x 4 x 4
    # fmt: off
    map_batch1 = torch.tensor(
            [
                [[1, 1, 0, 0],
                 [1, 1, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],

                [[1, 0, 0, 1],
                 [0, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 1, 1, 1]],

            ])
    map_batch1 = map_batch1.unsqueeze(0)
    map_batch2 = torch.tensor(
            [
                [[0, 0, 1, 1],
                 [0, 0, 1, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],

                [[0, 0, 0, 1],
                 [0, 0, 0, 0],
                 [1, 0, 0, 0],
                 [1, 1, 1, 1]],
            ]
    )
    map_batch2 = map_batch2.unsqueeze(0)

    output = mask_location_change(map_batch1, map_batch2)

    # Check the output
    expected_output = torch.tensor([1.0, 0.5])

    assert torch.allclose(output, expected_output)


if __name__ == "__main__":
    pytest.main()
