"""
This module contains helper functions mainly used for visualization purposes.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def invert_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    s = torch.tensor(np.asarray(std, dtype=np.float32)).unsqueeze(1).unsqueeze(2).cuda()
    m = (
        torch.tensor(np.asarray(mean, dtype=np.float32))
        .unsqueeze(1)
        .unsqueeze(2)
        .cuda()
    )

    res = img * s + m

    # check if image is in range [0, 1]
    res = torch.clamp(res, 0, 1)

    return res


def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    s = torch.tensor(np.asarray(std, dtype=np.float32)).unsqueeze(1).unsqueeze(2).cuda()
    m = (
        torch.tensor(np.asarray(mean, dtype=np.float32))
        .unsqueeze(1)
        .unsqueeze(2)
        .cuda()
    )

    res = (img - m) / s
    return res


def save_image_mask(img, saliency_map, file_path):
    """
    Save the image and the saliency map in the target directory.

    Args:
    image (torch.Tensor): The input image.
    saliency_map (torch.Tensor): The saliency map.
    target_dir (str): The target directory.
    """
    # invert the normalization from the dataloader
    img = invert_normalize(img).cpu()
    # change the channel order to (height, width, channels)
    img = img.squeeze(0).permute(1, 2, 0).numpy()

    fig, ax = plt.subplots()
    ax.imshow(img, alpha=0.5)
    overlay = ax.imshow(saliency_map.cpu(), cmap="viridis", alpha=0.5)

    # Add a colorbar
    # cbar = plt.colorbar(overlay, ax=ax)
    # cbar.set_label("Intensity")

    file_dir = os.path.dirname(file_path)
    os.makedirs(file_dir, exist_ok=True)

    # remove axis
    ax.axis("off")

    plt.savefig(file_path)
    # delete


def save_image_mask_bb(img, saliency_map, bounding_box, file_path):
    # invert the normalization from the dataloader
    img = invert_normalize(img).cpu()
    # change the channel order to (height, width, channels)
    img = img.squeeze(0).permute(1, 2, 0).numpy()

    fig, ax = plt.subplots()
    ax.imshow(img, alpha=0.5)
    overlay = ax.imshow(saliency_map.cpu(), cmap="viridis", alpha=0.5)

    # Add a colorbar
    # cbar = plt.colorbar(overlay, ax=ax)
    # cbar.set_label("Intensity")

    lower_y, upper_y, lower_x, upper_x = bounding_box
    rect = plt.Rectangle(
        (lower_x, lower_y),
        upper_x - lower_x,
        upper_y - lower_y,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)

    file_dir = os.path.dirname(file_path)
    os.makedirs(file_dir, exist_ok=True)

    # remove axis
    ax.axis("off")

    plt.savefig(file_path)


def save_image_bounding_box(img, bounding_box, file_path):
    """
    Save the image and the bounding box in the target directory.

    Args:
    image (torch.Tensor): The input image.
    bounding_box (tuple): The bounding box.
    target_dir (str): The target directory.
    """
    # if image is tensor
    if isinstance(img, torch.Tensor):
        # invert the normalization from the dataloader
        img = invert_normalize(img).cpu()
        # change the channel order to (height, width, channels)
        img = img.squeeze(0).permute(1, 2, 0).numpy()

    fig, ax = plt.subplots()
    ax.imshow(img)
    lower_x, lower_y, upper_x, upper_y = bounding_box
    rect = plt.Rectangle(
        (lower_x, lower_y),
        upper_x - lower_x,
        upper_y - lower_y,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)

    file_dir = os.path.dirname(file_path)
    os.makedirs(file_dir, exist_ok=True)

    # remove axis
    ax.axis("off")

    plt.savefig(file_path)


def save_mask(mask, file_path):
    """
    Save the mask in the target directory.

    Args:
    mask (torch.Tensor): The mask.
    target_dir (str): The target directory.
    """
    mask = mask.detach().cpu().numpy()
    plt.imshow(mask, cmap="viridis")

    file_dir = os.path.dirname(file_path)
    os.makedirs(file_dir, exist_ok=True)

    # remove axis
    plt.axis("off")

    plt.savefig(file_path)


def save_image(image, file_path):
    """
    Save the image in the target directory.

    Args:
    image (torch.Tensor): The input image.
    target_dir (str): The target directory.
    """
    image = invert_normalize(image).cpu()
    image = image.squeeze(0).permute(1, 2, 0).numpy()

    file_dir = os.path.dirname(file_path)
    os.makedirs(file_dir, exist_ok=True)

    # remove axis
    plt.axis("off")

    plt.imsave(file_path, image)


def save_image_bb_points(image, bb, points, file_path):

    image = invert_normalize(image).cpu()
    image = image.squeeze(0).permute(1, 2, 0).numpy()

    # bb are in the format (lower_y, upper_y, lower_x, upper_x)
    lower_y, upper_y, lower_x, upper_x = bb

    fig, ax = plt.subplots()
    ax.imshow(image)

    # add bb to the image
    rect = plt.Rectangle(
        (lower_x, lower_y),
        upper_x - lower_x,
        upper_y - lower_y,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)

    # add points to the image
    for point in points:
        y, x = point
        ax.scatter(x, y, c="r", s=10)

    file_dir = os.path.dirname(file_path)
    os.makedirs(file_dir, exist_ok=True)

    # remove axis
    ax.axis("off")

    plt.savefig(file_path)


# Good neon color (in BGR FROMAT): [58, 17, 203], [75, 5, 249]
def generate_neon_color() -> list[int]:
    """
    Generates a neon color (in BGR) to display the mask with.
    """
    blue = np.random.randint(0, 80)
    green = np.random.randint(0, 80)
    red = np.random.randint(200, 256)

    return [blue, green, red]


def show_mask_over_image(image_path, mask_path, save_path=None):
    """
    Display mask transparent over image. If multiple masks are provided,
    they will all be handled on the same image.

    Args:
    image_path (str): The image path.
    mask_path (str or list): The mask path or a list of mask paths.
    save_path (str): If given, the location to store the generated image.
    """
    image = cv2.imread(image_path)

    # If mask_path is a list, we need to iterate over it and process each mask
    if isinstance(mask_path, list):
        masks = [cv2.imread(mask, cv2.IMREAD_GRAYSCALE) for mask in mask_path]
    else:
        masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)]

    # Resize masks to match the image size, if needed
    for i in range(len(masks)):
        if image.shape[:2] != masks[i].shape:
            masks[i] = cv2.resize(masks[i], (image.shape[1], image.shape[0]))

        # Threshold mask to binary
        _, masks[i] = cv2.threshold(masks[i], 1, 255, cv2.THRESH_BINARY)
        masks[i] = (masks[i] > 0).astype(np.bool_)

    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    overlay = np.zeros_like(image_bgra, dtype=np.float32)

    # Use neon red for overlay color
    neon_red = [75, 5, 249]

    # Apply each mask to the overlay
    for i, mask in enumerate(masks):
        if i == 0:  # first mask is always neon_red
            overlay[mask] = np.concatenate([neon_red, [0.35]])
            continue
        # generate colors for all following masks
        random_color = generate_neon_color()
        overlay[mask] = np.concatenate([random_color, [0.35]])

    # Blend overlay with image
    alpha = overlay[..., 3]
    for c in range(3):
        image_bgra[..., c] = image_bgra[..., c] * (1 - alpha) + overlay[..., c] * alpha

    # Display image
    plt.imshow(cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGBA))
    plt.axis("off")

    # Save if save_path is provided
    if save_path:
        plt.savefig(save_path)

    plt.show()
