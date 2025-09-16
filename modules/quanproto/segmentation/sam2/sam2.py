from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# you have to first install the sam2 package
# checkout the github repository: https://github.com/facebookresearch/sam2
# or use this command from the repository:
# git clone https://github.com/facebookresearch/sam2.git && cd sam2
# pip install -e .

import os
import cv2
from quanproto.segmentation.best_params import sam2_download_paths, sam2_configs


def download_sam2_model(config):
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model_path = os.path.join(folder_path, config["checkpoint"])
    if not os.path.exists(model_path):
        os.system(f"wget {sam2_download_paths[config['model_type']]} -P {folder_path}")

    return model_path


def create_sam2_model(config):
    check_point = download_sam2_model(config)

    sam2 = build_sam2(
        sam2_configs[config["model_type"]],
        check_point,
        device="cuda",
        apply_postprocessing=False,
    )

    mask_gen_model = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=config["points_per_side"],
        points_per_batch=config["points_per_batch"],
        pred_iou_thresh=config["pred_iou_thresh"],
        stability_score_thresh=config["stability_score_thres"],
        stability_score_offset=config["stability_score_offset"],
        crop_n_layers=config["crop_n_layers"],
        mask_threshold=config["mask_threshold"],
        box_nms_thresh=config["box_nms_thresh"],
        crop_nms_thresh=config["crop_nms_thresh"],
        crop_overlap_ratio=config["crop_overlap_ratio"],
        crop_n_points_downscale_factor=config["crop_n_points_downscale_factor"],
        min_mask_region_area=config["min_mask_region_area"],
        use_m2m=config["use_m2m"],
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
