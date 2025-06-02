import pytest
import torch
from quanproto.metrics.continuity import (
    classification_activation_change,
    classification_rank_change,
    stability_score_bb,
    stability_score_mask,
)


def test_classification_activation_change_basic():
    # fmt: off

    # Create a batch of logits with 2 x 3
    logits1 = torch.tensor(
        [
            [0.1, 0.2, 1.0],
            [0.2, 0.4, 0.5],
        ]
    )

    logits2 = torch.tensor(
        [
            [0.2, 0.4, 0.5],
            [0.1, 0.2, 1.0],
        ]
    )

    # fmt: on

    # check the dimensions of the logits
    assert logits1.shape == (2, 3)
    assert logits2.shape == (2, 3)

    output = classification_activation_change(logits1, logits2)

    # Check the output
    expected_output = torch.tensor([0.5, 0.5])

    assert torch.allclose(output, expected_output)


def test_classification_rank_change_basic():
    # fmt: off

    # Create a batch of logits with 2 x 3
    logits1 = torch.tensor(
        [
            [0.1, 0.2, 1.0],
            [0.2, 0.4, 0.5],
        ]
    )

    logits2 = torch.tensor(
        [
            [1.0, 0.2, 0.1],
            [0.2, 0.4, 0.5],
        ]
    )

    # fmt: on

    output = classification_rank_change(logits1, logits2)

    # Check the output
    expected_output = torch.tensor([2, 0])

    assert torch.allclose(output, expected_output)


def test_stability_score_bb_basic():
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


    output = stability_score_bb(bbox_batch1, bbox_batch2, partlocs)

    # B x N x K
    expected_output = torch.tensor(
        [1,0, 0.5]
    ).float()

    assert torch.allclose(output, expected_output)
    # fmt: on


def test_stability_score_mask_basic():
    # fmt: off
    # map batch (1 x 2 x 7 x 7)
    map_batch1= torch.tensor(
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

    map_batch2= torch.tensor(
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
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,1,1,1,1,1,0],
             [0,1,1,1,1,1,0],
             [0,1,1,1,1,1,0]],

            [[0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0],
             [0,1,1,1,1,1,0],
             [0,1,1,1,1,1,0],
             [0,1,1,1,1,1,0]],
        ]]
    )
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


    output = stability_score_mask(map_batch1, map_batch2, partlocs)

    # B x N x K
    expected_output = torch.tensor(
        [1,0, 0.5]
    ).float()

    assert torch.allclose(output, expected_output)
    # fmt: on


if __name__ == "__main__":
    pytest.main()
