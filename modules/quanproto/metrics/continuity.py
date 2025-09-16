import torch


def classification_activation_change(logits1: torch.Tensor, logits2: torch.Tensor):
    """
    Compute the absolute difference between two activation maps

    Args:
        map_batch1: The first batch of activation maps (B x C)
        map_batch2: The second batch of activation maps (B x C)
    """
    # sort the activation maps
    sorted_logits1, _ = torch.sort(logits1, dim=1, descending=True)
    sorted_logits2, _ = torch.sort(logits2, dim=1, descending=True)

    intersection = torch.minimum(sorted_logits1, sorted_logits2).sum(dim=1)
    union = torch.maximum(sorted_logits1, sorted_logits2).sum(dim=1)

    change = 1 - intersection / union
    return change


def classification_rank_change(logits1: torch.Tensor, logits2: torch.Tensor):
    """
    Compute the relative change in the maximum activations

    Args:
        logits1: The first batch of logits (B x N)
        logits2: The second batch of logits (B x N)
    """

    # get the indices of the new rank
    _, ranks1 = torch.sort(logits1, dim=1, descending=True)
    _, ranks2 = torch.sort(logits2, dim=1, descending=True)

    # use the top 1 rank, because this is the predicted class
    top1_ranks = ranks1[:, 0]

    # expand the top 1 rank to the number of classes with the same rank
    expanded_top1_ranks = top1_ranks.unsqueeze(1).expand(-1, ranks2.shape[1])

    # compare the ranks
    compare_mask = expanded_top1_ranks == ranks2

    # get the new rank
    new_ranks = torch.argmax(compare_mask.int(), dim=1)
    return new_ranks


def stability_score_bb(bb_batch1: list, bb_batch2: list, partlocs: torch.Tensor):
    """
    Computes how many part locations are in the intersection of the bounding boxes
    and how many are in the union of the bounding boxes

    Args:
        bb_batch1: The first batch of bounding boxes (B x N x 4)
        bb_batch2: The second batch of bounding boxes (B x N x 4)
        partlocs: The part locations (B x K x 2)
    """

    if not isinstance(bb_batch1, torch.Tensor):
        bb_batch1 = torch.tensor(bb_batch1).to(partlocs.device)
    if not isinstance(bb_batch2, torch.Tensor):
        bb_batch2 = torch.tensor(bb_batch2).to(partlocs.device)

    assert bb_batch1.device == partlocs.device
    assert bb_batch2.device == partlocs.device

    # expand the bounding boxes to K so we have B x N x K x 4
    bb_batch1 = bb_batch1.unsqueeze(2).expand(-1, -1, partlocs.shape[1], -1)
    bb_batch2 = bb_batch2.unsqueeze(2).expand(-1, -1, partlocs.shape[1], -1)

    # expand the partlocs to N so we have B x N x K x 2
    partlocs = partlocs.unsqueeze(1).expand(-1, bb_batch1.shape[1], -1, -1)

    # compute the partlocs in bb1 and bb2
    in_bb1 = torch.logical_and(
        partlocs[:, :, :, 0] >= bb_batch1[:, :, :, 0],
        partlocs[:, :, :, 0] <= bb_batch1[:, :, :, 2],
    ) * torch.logical_and(
        partlocs[:, :, :, 1] >= bb_batch1[:, :, :, 1],
        partlocs[:, :, :, 1] <= bb_batch1[:, :, :, 3],
    )

    in_bb2 = torch.logical_and(
        partlocs[:, :, :, 0] >= bb_batch2[:, :, :, 0],
        partlocs[:, :, :, 0] <= bb_batch2[:, :, :, 2],
    ) * torch.logical_and(
        partlocs[:, :, :, 1] >= bb_batch2[:, :, :, 1],
        partlocs[:, :, :, 1] <= bb_batch2[:, :, :, 3],
    )

    # compute the union box
    in_union = torch.logical_or(in_bb1, in_bb2)
    num_partlocs = torch.sum(in_union.int(), dim=2)

    # compute the intersection box
    in_intersection = torch.logical_and(in_bb1, in_bb2)

    num_changed = torch.sum((in_union.int() - in_intersection.int()), dim=2)
    stability = 1 - num_changed.float() / num_partlocs.float()

    return stability


def stability_score_mask(map_batch1, map_batch2, partlocs):
    """
    Computes how many part locations are in the intersection of the maps
    and how many are in the union of the maps

    :param map_batch: The batch of activation maps (B x N x H x W)
    :type map_batch: torch.Tensor
    :param partlocs: The part locations (B x K x 2)
    :type partlocs: torch.Tensor
    """
    height_indices = partlocs[..., 1].long()
    width_indices = partlocs[..., 0].long()

    # B x K
    batch_indices = torch.arange(map_batch1.shape[0]).unsqueeze(1).expand(-1, partlocs.shape[1])
    # B x K x N
    in_map1 = map_batch1[batch_indices, :, height_indices, width_indices].int()
    in_map1 = in_map1.permute(0, 2, 1)  # B x N x K

    in_map2 = map_batch2[batch_indices, :, height_indices, width_indices].int()
    in_map2 = in_map2.permute(0, 2, 1)  # B x N x K

    in_union = torch.logical_or(in_map1, in_map2)
    num_partlocs = torch.sum(in_union.int(), dim=2)

    in_intersection = torch.logical_and(in_map1, in_map2)
    num_changed = torch.sum((in_union.int() - in_intersection.int()), dim=2)

    stability = 1 - num_changed.float() / num_partlocs.float()

    return stability
