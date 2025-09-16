# import quanproto.segmentation.sam.sam as sam
from quanproto.utils import parameter
from tqdm import tqdm
import requests
import numpy as np
import cv2
import os


def find_bounding_box(arr):
    """function to find the minimum bounding box for a given mask arr"""
    # Find the indices where the array is 1
    ones = np.where(arr == 1)

    if ones[0].size == 0:
        return None  # No ones in the array

    # Find the bounds of the bounding box
    top_left = (np.min(ones[0]), np.min(ones[1]))
    bottom_right = (np.max(ones[0]), np.max(ones[1]))

    return (top_left, bottom_right)


def cut_image_by_single_mask(img, mask):
    """cuts the mask out of an image and saves it

    Gets as input an img(.png or .jpeg) and saves the output in the target folder.
    If the mask is smaller than the min_ares it is skipped
    """
    return cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))


def cut_image_by_masks_and_save(img, masks, target_folder):
    """cuts the mask out of an image and saves it

    Gets as input an img(.png or .jpeg) and saves the output in the target folder.
    If the mask is smaller than the min_ares it is skipped
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(0, len(masks)):
        res = cv2.bitwise_and(img, img, mask=masks[i]["segmentation"].astype(np.uint8))
        cv2.imwrite(f"{target_folder}_{i}.jpg", res)


def shift_image(mask, shifts):
    M = np.float32([[1, 0, shifts[1]], [0, 1, shifts[0]]])
    shifted = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    return shifted


def compute_mask_center_shift(mask):
    """calculates the shift needed to put the mask in the center of the image
    shifts can be used like this:
    M = np.float32([[1, 0, shift_2], [0, 1, shift_1]])
    shifted = cv2.warpAffine(res, M, (image.shape[1], image.shape[0])
    """
    image_center = (int(mask.shape[0] / 2), int(mask.shape[1] / 2))
    top_left, bottom_right = find_bounding_box(mask)
    center_of_mask = (
        int((bottom_right[1] - top_left[1]) / 2 + top_left[1]),
        int((bottom_right[0] - top_left[0]) / 2 + top_left[0]),
    )
    shift_1 = image_center[0] - center_of_mask[1]
    shift_2 = image_center[1] - center_of_mask[0]
    return (shift_1, shift_2)


def compute_mask_shift_for_patch(top_left_patch, bottom_right_patch, org_img_size):
    image_center = (int(org_img_size[0] / 2), int(org_img_size[1] / 2))
    center_of_mask = (
        int((bottom_right_patch[1] - top_left_patch[1]) / 2 + top_left_patch[1]),
        int((bottom_right_patch[0] - top_left_patch[0]) / 2 + top_left_patch[0]),
    )
    shift_1 = image_center[0] - center_of_mask[1]
    shift_2 = image_center[1] - center_of_mask[0]
    return (-shift_1, -shift_2)


def compute_mask_size(mask, fraction=True):
    """calculates the size of the mask
    fraction: if True, returns the fraction of the mask
    """
    if fraction:
        return np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
    else:
        return np.count_nonzero(mask)


def check_and_download_sam(workspace_path, model_type="vit_h", target_folder="pretrained_models"):
    """
    Checks if a file at a given path already exists. If not, creates any necessary directories
    and downloads the file from the given URL with a progress bar.

    :param file_path: The path to check if the file exists.
    :param url: The URL to download the file from if it doesn't exist.
    """
    # sam_model_dir = "pretrained_models"
    file_path = os.path.join(workspace_path, target_folder)
    file_path = os.path.join(file_path, parameter.sam_download_paths[model_type].split("/")[-1])
    print(f"searching for file at path {file_path}")
    # Ensure all directories in the path exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directories: {directory}")

    if os.path.exists(file_path):
        # print(f"The file already exists at: {file_path}")
        pass
    else:
        try:
            url = parameter.sam_download_paths[model_type]
            print(f"File not found at {file_path}. Downloading from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for HTTP request errors

            # Get the total file size from the headers (if available)
            total_size = int(response.headers.get("content-length", 0))

            with open(file_path, "wb") as file, tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive new chunks
                        file.write(chunk)
                        progress_bar.update(len(chunk))

            print(f"Download complete. File saved at: {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during download: {e}")
