import torch
from torch.nn.functional import max_pool2d


def add_gaussian_noise(
    image: torch.Tensor, bb: tuple, mean: float = 0.0, std: float = 0.1
) -> torch.Tensor:
    """
    Add Gaussian noise to a tensor.
    """
    noise = torch.normal(mean, std, size=image.size()).to(image.device)

    # lower_y, upper_y, lower_x, upper_x = bb
    lower_x, lower_y, upper_x, upper_y = bb

    # overwrite the mask with the bounding box
    mask = torch.zeros(image.shape[-2:]).to(image.device)
    mask[lower_y : upper_y + 1, lower_x : upper_x + 1] = 1

    # delete noise where mask is 1
    noise = noise * (1 - mask)
    return image + noise


def gaussian_noise_mask(
    image: torch.Tensor, bb: tuple, mean: float = 0.0, std: float = 0.1
) -> torch.Tensor:
    noise = torch.normal(mean, std, size=image.size()).to(image.device)

    # lower_y, upper_y, lower_x, upper_x = bb
    lower_x, lower_y, upper_x, upper_y = bb

    # overwrite the mask with the bounding box
    mask = torch.ones(image.shape[-2:]).to(image.device)
    mask[lower_y : upper_y + 1, lower_x : upper_x + 1] = 0

    # delete noise where mask is 1
    noise = noise * mask
    return noise


def bb_location_change(bb_batch1: torch.Tensor | list, bb_batch2: torch.Tensor | list):
    """
    Compute the bounding box of the maps
    Compute 1 - intersection over union

    Args:
        bb_batch1: The first batch of bounding boxes (B x N x 4)
        bb_batch2: The second batch of bounding boxes (B x N x 4)
    """
    if not isinstance(bb_batch1, torch.Tensor):
        bb_batch1 = torch.tensor(bb_batch1)
    if not isinstance(bb_batch2, torch.Tensor):
        bb_batch2 = torch.tensor(bb_batch2)

    lower_x = torch.maximum(bb_batch1[:, :, 0], bb_batch2[:, :, 0])
    lower_y = torch.maximum(bb_batch1[:, :, 1], bb_batch2[:, :, 1])
    upper_x = torch.minimum(bb_batch1[:, :, 2], bb_batch2[:, :, 2])
    upper_y = torch.minimum(bb_batch1[:, :, 3], bb_batch2[:, :, 3])

    # intersection = torch.maximum(upper_y - lower_y, torch.tensor(0)) * torch.maximum(
    #     upper_x - lower_x, torch.tensor(0)
    # )
    intersection = (upper_y - lower_y) * (upper_x - lower_x)

    area1 = (bb_batch1[:, :, 2] - bb_batch1[:, :, 0]) * (
        bb_batch1[:, :, 3] - bb_batch1[:, :, 1]
    )

    area2 = (bb_batch2[:, :, 2] - bb_batch2[:, :, 0]) * (
        bb_batch2[:, :, 3] - bb_batch2[:, :, 1]
    )

    union = area1 + area2 - intersection

    overlap = intersection / union
    change = 1 - overlap
    return change


def activation_change(map_batch1: torch.Tensor, map_batch2: torch.Tensor):
    """
    Compute the absolute difference between two activation maps

    Args:
        map_batch1: The first batch of activation maps (B x N x H x W)
        map_batch2: The second batch of activation maps (B x N x H x W)
    """
    # flatten the activation maps
    flat_map1 = map_batch1.view(map_batch1.shape[0], map_batch1.shape[1], -1)
    flat_map2 = map_batch2.view(map_batch2.shape[0], map_batch2.shape[1], -1)

    # sort the activation maps
    flat_map1, _ = torch.sort(flat_map1, dim=2, descending=True)
    flat_map2, _ = torch.sort(flat_map2, dim=2, descending=True)

    intersection = torch.minimum(flat_map1, flat_map2).sum(dim=2)
    union = torch.maximum(flat_map1, flat_map2).sum(dim=2)

    change = 1 - intersection / union
    return change


def max_activation_location_change(map_batch1: torch.Tensor, map_batch2: torch.Tensor):
    """
    Compute the manhatten distance between the maximum activations

    Args:
        map_batch1: The first batch of activation maps (B x N x H x W)
        map_batch2: The second batch of activation maps (B x N x H x W)
    """

    # flatten the activation maps to (BxNx H*W)
    flat_map1 = map_batch1.view(map_batch1.shape[0], map_batch1.shape[1], -1)
    flat_map2 = map_batch2.view(map_batch2.shape[0], map_batch2.shape[1], -1)

    # find the maximum activations
    max_map1 = torch.argmax(flat_map1, dim=2)
    max_map2 = torch.argmax(flat_map2, dim=2)

    # unravel the index to get the coordinates
    max_map1 = torch.stack(
        [max_map1 // map_batch1.shape[2], max_map1 % map_batch1.shape[2]], dim=2
    )
    max_map2 = torch.stack(
        [max_map2 // map_batch2.shape[2], max_map2 % map_batch2.shape[2]], dim=2
    )

    # compute the manhatten distance
    distance = torch.abs(max_map1 - max_map2).sum(dim=2)
    return distance


def max_activation_change(map_batch1: torch.Tensor, map_batch2: torch.Tensor):
    """
    Compute the relative change in the maximum activations
    """
    max_map1 = max_pool2d(map_batch1, kernel_size=map_batch1.shape[2:])
    max_map2 = max_pool2d(map_batch2, kernel_size=map_batch2.shape[2:])

    change = torch.abs(max_map1 - max_map2) / max_map1
    return change


def rank_change(topk: torch.Tensor, scores: torch.Tensor):
    """
    Compute the rank change between two sets of scores
    TODO: Check correctness of rank.
    If scores.shape is (B x k x N) the rank change is calculated for all k prototypes.
    If scores.shape is (B x N) the rang change is calculated for only one prototype,
    but scores is expanded to (B x k x N) with k equal vectors.

    Args:
        topk: The top k indices (B x K)
        scores: The new scores (B x N) or (B x k x N)
    """

    # (B x N) -> (B x K x N)
    if len(scores.shape) == 2:
        scores = scores.unsqueeze(1).expand(-1, topk.shape[1], -1)

    # get the indices of the new rank
    _, ranks = torch.sort(scores, dim=2, descending=True)

    topk_expanded = topk.unsqueeze(2).expand_as(ranks)

    compare_mask = topk_expanded == ranks

    new_ranks = torch.argmax(compare_mask.int(), dim=2)

    # make a tensor with Bx [0, 1, 2, ..., K-1]
    old_rank = torch.arange(topk.shape[1]).unsqueeze(0).expand_as(topk).to(topk.device)

    # compute the rank change
    change = torch.abs(old_rank - new_ranks)
    return change


def mask_location_change(map_batch1: torch.Tensor, map_batch2: torch.Tensor):
    """
    Compute 1 - intersection over union

    Args:
        map_batch1: The first batch of activation maps (B x N x H x W)
        map_batch2: The second batch of activation maps (B x N x H x W)
    """
    intersection = torch.logical_and(map_batch1, map_batch2).sum(dim=(2, 3))
    union = torch.logical_or(map_batch1, map_batch2).sum(dim=(2, 3))
    overlap = intersection / union
    change = 1 - overlap
    return change
