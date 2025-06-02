from typing import Callable

import numpy as np
import torch


def label_prediction(logits: torch.Tensor, multi_label=False, threshold=0.5):

    if multi_label:
        # use the threshold
        logits = logits > threshold

    else:
        # softmax
        logits = torch.softmax(logits, dim=1)
        logits = torch.argmax(logits, dim=1)

    return logits


def percentile_mask(map: torch.Tensor, percentile: float = 95.0) -> torch.Tensor:
    """
    Compute a binary mask based on the given percentile of the activation map
    """
    # Compute the threshold based on the given percentile
    threshold = torch.quantile(map, percentile / 100.0)

    # Create a mask with ones where activation values are above the threshold
    mask = torch.ones_like(map)
    mask[map < threshold] = 0

    return mask


def binary_mask(map) -> torch.Tensor:
    """
    Compute a binary mask
    """
    if isinstance(map, np.ndarray):
        map = torch.from_numpy(map)

    mask = torch.ones_like(map)
    mask[map <= 0.0] = 0

    return mask


def min_max_norm_mask(map: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Compute a binary mask based on a min max normalization with a threshold
    """
    # Compute the minimum and maximum values of the activation map
    min_val = torch.amin(map)
    max_val = torch.amax(map)

    # Normalize the activation map to the range [0, 1]
    norm_map = (map - min_val) / (max_val - min_val)

    # use a threshold of 0.5 to create a binary mask
    mask = torch.where(norm_map > threshold, torch.tensor(1.0), torch.tensor(0.0))

    return mask


def bounding_box(
    map: torch.Tensor, mask_fn: Callable = percentile_mask, **kwargs
) -> tuple:
    """
    Compute the bounding box of the activation map

    Args:
        map: The activation map (H x W)
        mask_fn: The function to compute the mask (default: percentile_mask)
        **kwargs: Additional arguments for the mask function

    Returns:
        The bounding box coordinates (lower_x, lower_y, upper_x, upper_y)
    """
    # Compute the mask based on the given function
    mask = mask_fn(map, **kwargs)

    # Initialize the crop coordinates
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0

    # Find the lower and upper y bounds
    for i in range(mask.shape[0]):
        if torch.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if torch.amax(mask[i]) > 0.5:
            upper_y = i
            break

    # Find the lower and upper x bounds
    for j in range(mask.shape[1]):
        if torch.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if torch.amax(mask[:, j]) > 0.5:
            upper_x = j
            break

    return lower_x, lower_y, upper_x, upper_y


def bounding_box_mask(
    map: torch.Tensor, mask_fn: Callable = percentile_mask, **kwargs
) -> tuple:

    lower_x, lower_y, upper_x, upper_y = bounding_box(map, mask_fn, **kwargs)

    # overwrite the mask with the bounding box
    mask = torch.zeros_like(map)
    mask[lower_y:upper_y, lower_x:upper_x] = 1

    return mask
