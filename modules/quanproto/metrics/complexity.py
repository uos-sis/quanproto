import torch


def mask_intersection_over_union(map_batch: torch.Tensor, mask_batch: torch.Tensor):
    """
    Compute intersection over union with a batch of saliency maps and a batch of ground truth masks

    Args:
        map_batch: The first batch of activation maps (B x N x H x W)
        mask_batch: The second batch of activation maps (B x H x W)
    """
    # expand the mask to the number of maps
    mask_batch = mask_batch.unsqueeze(1).repeat(1, map_batch.shape[1], 1, 1)

    intersection = torch.logical_and(map_batch, mask_batch).sum(dim=(2, 3))
    union = torch.logical_or(map_batch, mask_batch).sum(dim=(2, 3))
    overlap = intersection / union

    # if union is zero, the overlap is 0
    overlap[union == 0] = 0

    return overlap


def mask_overlap(map_batch: torch.Tensor, mask_batch: torch.Tensor):
    """
    Compute the overlap of the map with the mask

    Args:
        map_batch: The first batch of activation maps (B x N x H x W)
        mask_batch: The batch of segmentation masks (B x H x W)
    """
    # check the dimensions
    assert (
        map_batch.shape[0] == mask_batch.shape[0]
    ), f"map batch = {map_batch.shape} != mask batch = {mask_batch.shape}"
    assert (
        map_batch.shape[2] == mask_batch.shape[1]
    ), f"map batch = {map_batch.shape} != mask batch = {mask_batch.shape}"
    assert (
        map_batch.shape[3] == mask_batch.shape[2]
    ), f"map batch = {map_batch.shape} != mask batch    = {mask_batch.shape}"

    # expand the mask to the number of maps
    mask_batch = mask_batch.unsqueeze(1).repeat(1, map_batch.shape[1], 1, 1)

    intersection = torch.logical_and(map_batch, mask_batch).sum(dim=(2, 3))
    mask = mask_batch.sum(dim=(2, 3))
    overlap = intersection / mask

    # if mask is zero, the overlap is 0
    overlap[mask == 0] = 0

    return overlap


def background_overlap(map_batch: torch.Tensor, mask_batch: torch.Tensor):
    """Compute the overlap of the map with the background

    :param map_batch: batch of activation maps (B x N x H x W)
    :type map_batch: torch.Tensor
    :param mask_batch: batch of segmentation masks (B x H x W)
    :type mask_batch: torch.Tensor
    """
    # expand the mask to the number of maps
    mask_batch = mask_batch.unsqueeze(1).repeat(1, map_batch.shape[1], 1, 1)

    intersection = torch.logical_and(map_batch, mask_batch).sum(dim=(2, 3))
    # binerize the map before summing
    map_batch = (map_batch > 0).int()
    map_size = map_batch.sum(dim=(2, 3))

    background_overlap = 1 - (intersection / map_size)

    # if map_size is zero, the background overlap is 0
    background_overlap[map_size == 0] = 0

    return background_overlap


def outside_inside_relevance_ratio(
    activation_map_batch: torch.Tensor, mask_batch: torch.Tensor
):
    """
    Compute the ratio of the relevance outside the mask to the relevance inside the mask

    Args:
    activation_map_batch: The batch of activation maps (B x N x H x W)
    mask_batch: The batch of segmentation masks (B x H x W)
    """
    epsilon = 1e-8
    # expand the mask to the number of maps
    mask_batch = mask_batch.unsqueeze(1).repeat(1, activation_map_batch.shape[1], 1, 1)

    outside = activation_map_batch * (mask_batch.int() - 1) * -1
    num_outside = torch.count_nonzero(outside, dim=(2, 3))

    inside = activation_map_batch * mask_batch.int()
    num_inside = torch.count_nonzero(inside, dim=(2, 3))

    mean_outside_activation = torch.sum(outside, dim=(2, 3)) / num_outside

    mean_inside_activation = torch.sum(inside, dim=(2, 3)) / num_inside

    torch.where(mean_inside_activation > epsilon, epsilon, mean_inside_activation)

    ratio = mean_outside_activation / mean_inside_activation

    # if num_inside is zero, the ratio is infinite
    ratio[num_inside == 0] = float("inf")

    # if only inside activations are present, the ratio is 0
    ratio[torch.logical_and(num_inside != 0, num_outside == 0)] = 0

    return ratio


def inside_outside_relevance(
    activation_map_batch: torch.Tensor, mask_batch: torch.Tensor
):
    """
    Compute the mean difference of the relevance inside the mask to the relevance outside the mask
    This metric is more stable than the ratio metrics and is well defined between -1 and 1
    negative means that the relevance is more outside the mask
    0 means that the relevance is equally distributed
    positive means that the relevance is more inside the mask

    A perfect score of 1 means that the mean inside relevance is also the maximum value of the map
    and the outside relevance is zero

    A perfect score of -1 means that the mean outside relevance is also the maximum value of the map
    and the inside relevance is zero


    Args:
    activation_map_batch: The batch of activation maps (B x N x H x W)
    mask_batch: The batch of segmentation masks (B x H x W)
    """
    # normalize the mask, get the max values of the two last dimensions
    max_values = torch.max(activation_map_batch, dim=3)[0]
    max_values = torch.max(max_values, dim=2)[0]  # B x N

    activation_map_batch = activation_map_batch / max_values.unsqueeze(2).unsqueeze(3)
    # if max val is zero, the map is zero
    activation_map_batch[max_values == 0] = 0

    # expand the mask to the number of maps
    mask_batch = mask_batch.unsqueeze(1).repeat(1, activation_map_batch.shape[1], 1, 1)

    outside = activation_map_batch * (mask_batch.int() - 1) * -1
    num_outside = torch.count_nonzero(outside, dim=(2, 3))

    inside = activation_map_batch * mask_batch.int()
    num_inside = torch.count_nonzero(inside, dim=(2, 3))

    mean_outside_activation = torch.sum(outside, dim=(2, 3)) / num_outside
    # if num_outside is zero, the mean is zero
    mean_outside_activation[num_outside == 0] = 0

    mean_inside_activation = torch.sum(inside, dim=(2, 3)) / num_inside
    # if num_inside is zero, the mean is zero
    mean_inside_activation[num_inside == 0] = 0

    difference = mean_inside_activation - mean_outside_activation

    return difference


def boundingbox_consistency(
    bb_batch: torch.Tensor | list, partlocs: torch.Tensor, partlocs_ids: torch.Tensor
):
    """
    Computes how many part locations are in the bounding box
    returns a tensor with the ids

    Args:
        bb_batch: The batch of bounding boxes (B x N x [])
        partlocs: The part locations (B x K x 2)
        partlocs_ids: The part locations ids (B x K)

    Returns:
        The ids of the part locations that are in the bounding box (B x N x K)
    """
    if not isinstance(bb_batch, torch.Tensor):
        bb_batch = torch.tensor(bb_batch)

    # check if bb_batch is on the same device as the partlocs
    if bb_batch.device != partlocs.device:
        bb_batch = bb_batch.to(partlocs.device)

    # expand the bounding boxes to K so we have B x N x K x 4
    bb_batch = bb_batch.unsqueeze(2).expand(-1, -1, partlocs.shape[1], -1)

    # expand the partlocs to N so we have B x N x K x 2
    partlocs = partlocs.unsqueeze(1).expand(-1, bb_batch.shape[1], -1, -1)
    partlocs_ids = partlocs_ids.unsqueeze(1).expand(-1, bb_batch.shape[1], -1)

    in_bb = torch.logical_and(
        partlocs[:, :, :, 0] >= bb_batch[:, :, :, 0],
        partlocs[:, :, :, 0] <= bb_batch[:, :, :, 2],
    ) * torch.logical_and(
        partlocs[:, :, :, 1] >= bb_batch[:, :, :, 1],
        partlocs[:, :, :, 1] <= bb_batch[:, :, :, 3],
    )

    # remove the partloc ids that are not in the bounding box
    partlocs_ids = partlocs_ids * in_bb.int()

    return partlocs_ids


def map_consistency(
    map_batch: torch.Tensor, partlocs: torch.Tensor, partlocs_ids: torch.Tensor
):
    """Compute how many part locations are in the map

    :param map_batch: The batch of activation maps (B x N x H x W)
    :type map_batch: torch.Tensor
    :param partlocs: The part locations (B x K x 2)
    :type partlocs: torch.Tensor
    :param partlocs_ids: The part locations ids (B x K)
    :type partlocs_ids: torch.Tensor
    """
    height_indices = partlocs[..., 1].long()
    width_indices = partlocs[..., 0].long()

    # B x K
    batch_indices = (
        torch.arange(map_batch.shape[0]).unsqueeze(1).expand(-1, partlocs.shape[1])
    )
    # B x K x N
    in_map = map_batch[batch_indices, :, height_indices, width_indices].int()
    in_map = in_map.permute(0, 2, 1)  # B x N x K

    # remove the partloc ids that are not in the map
    partlocs_ids = partlocs_ids.unsqueeze(1).expand(-1, map_batch.shape[1], -1) * in_map

    return partlocs_ids
