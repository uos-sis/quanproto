import torch
import skimage as ski
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import os
import itertools

from quanproto.metrics import complexity
import multiprocessing
import concurrent.futures
import albumentations as A
from torch.utils.data import Dataset, DataLoader


def load_data(mask_root, mask_paths, partlocs_info, max_partlocs):
    trans_pip = A.Compose(
        [
            A.Resize(224, 224),  # so we can make a tensor of the masks
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy"),
    )

    tf_masks = torch.empty(0)
    # use all available masks
    num_masks = len(mask_paths)

    for i in range(num_masks):
        mask_path = mask_paths[i]
        original_mask = ski.io.imread(os.path.join(mask_root, mask_path), as_gray=True)

        if len(original_mask.shape) == 3:
            if original_mask.shape[2] == 4:
                original_mask = (ski.color.rgba2rgb * 255).astype(np.uint8)
                original_mask = ski.color.rgb2gray(original_mask)
            elif original_mask.shape[2] == 2:
                original_mask = original_mask[:, :, 0]
            else:
                original_mask = ski.color.rgb2gray(original_mask)

        assert len(original_mask.shape) == 2, "Mask is not 2D"
        original_mask = original_mask > 0
        original_mask = original_mask.astype(np.uint8)

        height, width = original_mask.shape

        # partlocs = [p for p in partlocs_info.values()]
        partlocs = [
            (p[0] if p[0] < width else width - 1, p[1] if p[1] < height else height - 1)
            for p in partlocs_info.values()
        ]
        partlocs_ids = [p for p in partlocs_info.keys()]

        for i, point in enumerate(partlocs):
            # check if the point is within the image bounds
            if point[0] < 0 or point[0] >= width or point[1] < 0 or point[1] >= height:
                print(f"Mask path: {mask_path}")
                raise ValueError(
                    f"Partloc {point} is out of bounds for image of size {height}x{width}"
                )

        # first transform the keypoints based on the image
        t = trans_pip(image=original_mask, keypoints=partlocs)
        tf_partlocs = t["keypoints"]
        tf_mask = t["image"]

        # check what dtype is returned
        if isinstance(tf_mask, torch.Tensor):
            tf_mask = tf_mask.float()
        else:
            tf_mask = torch.tensor(tf_mask).float()
        tf_masks = torch.cat((tf_masks, tf_mask.unsqueeze(0)), 0)

    # make a tensor with (max_partlocs, 2) and fill it with -1
    t_partlocs = torch.full((max_partlocs, 2), -1, dtype=torch.float32)
    t_partlocs_ids = torch.full((max_partlocs, 1), -1, dtype=torch.int32)

    # overwrite the -1 entries with the actual partlocs
    for i, point in enumerate(tf_partlocs):
        t_partlocs[i] = torch.tensor([point[0], point[1]], dtype=torch.float32)
        t_partlocs_ids[i] = torch.tensor(partlocs_ids[i], dtype=torch.int32)

    return (
        tf_masks,
        t_partlocs,
        t_partlocs_ids,
    )


def evaluate_mask_consistency_kernel(args):
    mask_root, mask_paths, partlocs_info, max_partlocs, label = args
    # get the data
    masks, partlocs, partloc_ids = load_data(
        mask_root,
        mask_paths,
        partlocs_info,
        max_partlocs,
    )

    if len(masks.shape) == 4:
        # add the batch dimension
        masks = masks.unsqueeze(0)
        partlocs = partlocs.unsqueeze(0)
        partloc_ids = partloc_ids.unsqueeze(0)
        label = label.unsqueeze(0)

    # the masks should be of shape (B x N x 1 x H x W)
    # make it (B x N x H x W)
    masks = masks.squeeze(2)

    # the partloc ids should be of shape (B x N x 1)
    # make it (B x N)
    partloc_ids = partloc_ids.squeeze(2)

    overlap_ids = complexity.mask_consistency(
        masks,
        partlocs,
        partloc_ids,
    )

    return label.item(), overlap_ids


def evaluate_mask_consistency(data_obj):

    if isinstance(data_obj, Dataset):
        dataset = data_obj
        num_classes = dataset.num_classes
        num_partlocs = dataset.max_partlocs
    elif isinstance(data_obj, DataLoader):
        dataloader = data_obj
        num_classes = dataloader.dataset.num_classes
        num_partlocs = dataloader.dataset.max_partlocs
    else:
        raise ValueError("data_obj must be a Dataset or DataLoader")

    max_id_number = 2**num_partlocs
    # make a tensor with num_classes x max_id_number
    consistency_tensor = torch.zeros(
        (num_classes, max_id_number),
        dtype=torch.int32,
    )
    update_counter = torch.zeros(num_classes, dtype=torch.int32)

    if isinstance(data_obj, Dataset):
        for i in tqdm(range(len(dataset)), desc="Evaluating mask consistency"):
            args = dataset.get_item_params(i)
            result = evaluate_mask_consistency_kernel(args)
            consistency_tensor[result[0], result[1]] += 1
            update_counter[result[0]] += 1

    if isinstance(data_obj, DataLoader):
        data_iter = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Mask Consistency Evaluation",
            mininterval=2.0,
            ncols=0,
        )

        for i, (masks, partlocs, partloc_ids, label) in data_iter:
            # the masks should be of shape (B x N x 1 x H x W)
            # make it (B x N x H x W)
            masks = masks.squeeze(2)

            # the partloc ids should be of shape (B x N x 1)
            # make it (B x N)
            partloc_ids = partloc_ids.squeeze(2)

            overlap_ids = complexity.mask_consistency(
                masks,
                partlocs,
                partloc_ids,
            )

            # add the overlap ids to the consistency tensor
            consistency_tensor[label.item(), overlap_ids] += 1
            update_counter[label.item()] += 1

    class_consistency_histograms = []
    for i in range(num_classes):
        # remove all zero entries from the histogram
        histogram = consistency_tensor[i]
        histogram = histogram[histogram != 0]
        # normalize the histogram
        histogram = histogram.float() / update_counter[i].float()
        # convert to numpy
        histogram = histogram.cpu().numpy()
        class_consistency_histograms.append(histogram)

    return class_consistency_histograms
