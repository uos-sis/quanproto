import torch
import quanproto.segmentation.best_params as best_params
import numpy as np
import random


def slice_image(img, bbox):
    # make a tensor from the numpy array with dim (H, W, C)
    height, width = img.shape[0], img.shape[1]

    assert len(bbox) == 4, "Bounding box must have 4 elements"

    min_x = bbox[0].item()
    min_y = bbox[1].item()
    max_x = bbox[2].item()
    max_y = bbox[3].item()

    # TODO: use the bounding box shape to create the patch size
    patch_size_x = int((max_x - min_x) / best_params.patch_params["split_factor"])
    patch_size_y = int((max_y - min_y) / best_params.patch_params["split_factor"])
    num_patches = best_params.patch_params["split_factor"] ** 2

    # TODO: use the bounding box min point instead of 0
    x_coords, y_coords = torch.meshgrid(
        torch.arange(
            min_x, min_x + best_params.patch_params["split_factor"] * patch_size_x, patch_size_x
        ),
        torch.arange(
            min_y, min_y + best_params.patch_params["split_factor"] * patch_size_y, patch_size_y
        ),
        indexing="ij",
    )

    mask_list = []
    for i in range(0, num_patches):
        mask = np.zeros((height, width))

        start_x = int(tuple(x_coords.flatten())[i])
        start_y = int(tuple(y_coords.flatten())[i])

        end_x = start_x + patch_size_x
        end_y = start_y + patch_size_y

        if end_x > width:
            end_x = width
        if end_y > height:
            end_y = height

        assert start_x < end_x, "Start x must be smaller than end x"
        assert start_y < end_y, "Start y must be smaller than end y"

        mask[start_y:end_y, start_x:end_x] = 1
        mask_list.append(mask)

    # shuffle the mask list
    random.shuffle(mask_list)

    return mask_list


def create_patches_model():
    return slice_image


def segment_image(model, img, bbox):

    mask_list = model(img, bbox)
    return mask_list


if __name__ == "__main__":
    img_path = (
        "/home/pschlinge/data/samxproto/dogs/samples/n02105855-Shetland_sheepdog/n02105855_2933.jpg"
    )

    import skimage as ski
    import cv2
    import os
    import quanproto.segmentation.helpers as helpers

    img = ski.io.imread(img_path)
    if len(img.shape) == 2:
        # convert to 3 channels
        print("gray image")
        img = ski.color.gray2rgb(img)
    if img.shape[2] == 4:
        print("rgba image")
        img = (ski.color.rgba2rgb(img) * 255).astype(np.uint8)
        print(img.shape)
    if img.shape[2] == 2:
        raise ValueError("Image has 2 channels")

    img = cv2.resize(img, (224, 224))

    masks = segment_image(create_patches_model(), img)
    masks = [helpers.cut_image_by_single_mask(img, mask) for mask in masks]

    # save masks to disk
    path = "/home/pschlinge/data/tmp_masks"
    if os.path.exists(path):
        os.system(f"rm -r {path}")
    os.makedirs(path, exist_ok=True)

    for i, mask in enumerate(masks):
        ski.io.imsave(f"{path}/mask_{i}.png", mask)
