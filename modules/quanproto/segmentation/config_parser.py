import quanproto.segmentation.sam.sam as sam
import quanproto.segmentation.sam2.sam2 as sam2
import quanproto.segmentation.slit.slit as slit
import quanproto.segmentation.patches.patches as patches
import quanproto.segmentation.best_params as best_params

segmentation_method_fn = {
    "sam": sam.segment_image,
    "sam2": sam2.segment_image,
    "slit": slit.segment_image,
    "patches": patches.segment_image,
}

segmentation_model_fn = {
    "sam": sam.create_sam_model,
    "sam2": sam2.create_sam2_model,
    "slit": slit.create_slit_model,
    "patches": patches.create_patches_model,
}

model_config_dic = {
    "sam": best_params.sam_params,
    "sam2": best_params.sam2_params,
    "slit": best_params.slit_params,
    "patches": best_params.patch_params,
}

# you can skip the first n masks because they are often the background
skip_idx = {
    "sam": 0,
    "sam2": 0,
    "slit": 0,
    "patches": 0,
}
