"""
This module contains helper functions mainly used for visualization purposes.
"""

import torch
import torch.nn as nn


def model_output_size(
    model: nn.Module, input_size: tuple[int, int, int], tuple_idx: int = 0
) -> int | tuple[int, int, int]:
    """
    Compute the output size of a model.

    Args:
    model (torch.nn.Module): A PyTorch model.
    input (torch.Tensor): The input tensor.

    Returns:
    The output size (channels, height, width).
    """
    dummy_input = torch.rand(1, *input_size)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        if len(output) == 2:
            output = output[tuple_idx]

    if tuple_idx == 1:
        return output.size(1)

    return (output.size(1), output.size(2), output.size(3))


def convert_to_multilabelmargin_input(target):
    gpu_index_label = (
        torch.arange(target.size(1)).unsqueeze(0).expand(target.size(0), -1).cuda()
    )
    # replace values that are not in the gpu_label mask with -1
    gpu_index_label = torch.where(
        target == 0, -1 * torch.ones_like(target).cuda(), gpu_index_label
    )
    # sort in reverse order
    gpu_index_label, _ = torch.sort(gpu_index_label, dim=1, descending=True)

    # convert to long
    gpu_index_label = gpu_index_label.long()
    return gpu_index_label


def get_min_dist_vectors(
    dist_batch: torch.Tensor, feature_map: torch.Tensor, batch_min_idx
):
    """
    Find the feature vectors of the min distance

    Args:
        dist_batch: The first batch of activation maps (B x N x H x W)
        feature_map: The feature map of the model (B x D x H x W)
        batch_min_idx: The index of the minimum in batch (N)
    """

    # flatten the dist maps to (BxNx H*W)
    flat_map = dist_batch.view(dist_batch.shape[0], dist_batch.shape[1], -1)

    # flatten the feature map to (BxD H*W)
    flat_feature = feature_map.view(feature_map.shape[0], feature_map.shape[1], -1)

    # find the min index
    min_map = torch.argmin(flat_map, dim=2)  # BxN
    # min_map = min_map[batch_min_idx, :]  # N

    # make an index array with batch_min_idx size
    proto_idx = torch.arange(batch_min_idx.size(0)).cuda()

    min_map = min_map[batch_min_idx, proto_idx]  # N

    features = flat_feature[batch_min_idx, :, min_map].unsqueeze(-1).unsqueeze(-1)

    return features
