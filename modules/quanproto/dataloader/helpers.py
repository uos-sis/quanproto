import numpy as np

from quanproto.utils.vis_helper import save_image


def add_mean_background(mask: np.ndarray) -> np.ndarray:
    """
    Add a mean background to the mask.

    Args:
        mask (np.ndarray): [H, W, C] array
    """
    # Calculate one-hot representation of mask, assuming non-zero values are considered "foreground"
    one_hot = np.sum(mask, axis=-1) > 0
    size = np.sum(one_hot)

    if size == 0:
        return mask

    # Calculate mean for each channel (assuming mask is in [H, W, C] format)
    mean_channel_0 = int(np.sum(mask[:, :, 0]) / size)
    mean_channel_1 = int(np.sum(mask[:, :, 1]) / size)
    mean_channel_2 = int(np.sum(mask[:, :, 2]) / size)

    # Create background with the mean values
    background = np.zeros_like(mask).astype(np.uint8)
    # cast to uint8
    background[:, :, 0] = mean_channel_0
    background[:, :, 1] = mean_channel_1
    background[:, :, 2] = mean_channel_2

    # Invert one-hot mask and apply it to the background
    one_hot = (one_hot.astype(int) * -1 + 1).astype(np.uint8)
    background = background * np.expand_dims(one_hot, axis=-1)

    # Add the background to the original mask
    mask = (mask + background).astype(np.uint8)

    return mask


def add_original_background(mask: np.ndarray, img: np.ndarray):
    # Calculate one-hot representation of mask, assuming non-zero values are considered "foreground"
    one_hot = np.sum(mask, axis=-1) > 0
    one_hot = (one_hot.astype(int) * -1 + 1).astype(np.uint8)
    background = img * np.expand_dims(one_hot, axis=-1)
    mask = (mask + background).astype(np.uint8)

    return mask


def expand_bboxes(bboxes, img_size, min_percent: float = 0.1):
    """
    Expand the bounding boxes if it is to small.
    Args:
        bboxes (list): List of bounding boxes.
        img_size (tuple): (width, height) of the image.
        min_percent (float): Minimum percentage of the image the bounding box should cover.
    """
    expanded_bboxes = []
    # check if bbox is a list(list) or a single bbox
    if isinstance(bboxes, list) and not isinstance(bboxes[0], list):
        bboxes = [bboxes]

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        img_width, img_height = img_size

        # Calculate the area of the bounding box
        area = width * height
        # Calculate the area of the image
        img_area = img_width * img_height

        # Calculate the percentage of the image the bounding box covers
        percent = area / img_area
        if percent < min_percent:
            # Expand the bounding box to cover the minimum percentage
            new_width = int(img_width * min_percent)
            new_height = int(img_height * min_percent)

            x1 = max(0, x1 - (new_width - width) // 2)
            y1 = max(0, y1 - (new_height - height) // 2)
            x2 = min(img_width, x1 + new_width)
            y2 = min(img_height, y1 + new_height)
        expanded_bboxes.append((x1, y1, x2, y2))
    return expanded_bboxes
