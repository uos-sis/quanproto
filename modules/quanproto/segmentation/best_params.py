sam_params = {
    "method": "sam",
    "checkpoint": "sam_vit_h_4b8939.pth",
    "model_type": "vit_h",
    "model_version": "sam_vit_h_4b8939.pth",
    "points_per_side": 32,  # 32,
    "pred_iou_thresh": 0.86,  # 0.86,
    "stability_score_thresh": 0.92,  # 0.92,
    "crop_n_layers": 1,  # 1
    "crop_n_points_downscale_factor": 2,  # 2
    # "min_mask_area": 100,  # 100
}
sam_download_paths = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
}

sam2_params = {
    "method": "sam2",
    "checkpoint": "sam2.1_hiera_base_plus.pt",
    "model_type": "sam_b+",
    "points_per_side": 32,  # 32
    "points_per_batch": 512,  # 64
    "pred_iou_thresh": 0.8,  # 0.8
    "stability_score_thres": 0.9,  # 0.95
    "stability_score_offset": 0.5,  # 1.0
    "crop_n_layers": 2,  # 0
    "mask_threshold": 0.0,  # 0.0
    "box_nms_thresh": 0.7,  # 0.7
    "crop_nms_thresh": 0.7,  # 0.7
    "crop_overlap_ratio": 512 / 1500,  # 512 / 1500
    "crop_n_points_downscale_factor": 1,  # 1
    "min_mask_region_area": 10.0,  # 25.0
    "use_m2m": False,  # False
}

sam2_download_paths = {
    "sam_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam_small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam_b+": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam_large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}
sam2_configs = {
    "sam_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam_b+": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

slit_params = {
    "method": "slit",
    "threshold": 0.001,
    "checkpoint": "chendelong/DirectSAM-1800px-0424",
}

patch_params = {
    "method": "patches",
    "split_factor": 2,
}
