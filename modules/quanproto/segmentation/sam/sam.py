from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import os
import cv2
from quanproto.segmentation.best_params import sam_download_paths


def download_sam_model(sam_config):
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model_path = os.path.join(folder_path, sam_config["model_version"])
    if not os.path.exists(model_path):
        os.system(f"wget {sam_download_paths[sam_config['model_type']]} -P {folder_path}")

    return model_path


def create_sam_model(sam_config):
    check_point = download_sam_model(sam_config)
    sam = sam_model_registry[sam_config["model_type"]](checkpoint=check_point)
    sam.to(device="cuda")

    mask_gen_model = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=sam_config["points_per_side"],  # 32,
        pred_iou_thresh=sam_config["pred_iou_thresh"],  # 0.86,
        stability_score_thresh=sam_config["stability_score_thresh"],  # 0.92,
        crop_n_layers=sam_config["crop_n_layers"],  # 1
        crop_n_points_downscale_factor=sam_config["crop_n_points_downscale_factor"],  # 2
        # min_mask_region_area=sam_config["min_mask_area"],  # 100
    )
    return mask_gen_model


def segment_image(model, img, bbox):
    # convert the image to opencv format
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    sam_mask_list = model.generate(img)

    return_list = []
    for i in range(0, len(sam_mask_list)):
        return_list.append(sam_mask_list[i]["segmentation"])

    return return_list
