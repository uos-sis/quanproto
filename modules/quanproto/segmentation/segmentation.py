import skimage as ski
import os
import quanproto.segmentation.helpers as helpers
import quanproto.segmentation.config_parser as config_parser
from quanproto.metrics import helpers as metric_helpers
import cv2


def generate_segment_masks(dataset, dataloader, model_config):
    segmentation_method = config_parser.segmentation_method_fn[model_config["method"]]
    segmentation_model = config_parser.segmentation_model_fn[model_config["method"]](model_config)

    # delete the old segmentation masks from the dataset
    dataset.delete_segmentation_masks(model_config["method"])

    # iterate through the dataloader
    # batch size is 1 because we have different image sizes
    for img, bbox, idx in dataloader:
        img = img[0].numpy()
        img_idx = idx[0].item()

        masks = segmentation_method(segmentation_model, img, bbox)
        assert len(masks) > 0, "No masks found"

        # region compute additional information --------------------------------
        shifts = []
        sizes = []
        bounding_boxes = []
        for mask in masks:
            shifts.append(helpers.compute_mask_center_shift(mask))
            sizes.append(helpers.compute_mask_size(mask))
            bounding_boxes.append(
                metric_helpers.bounding_box(mask, mask_fn=metric_helpers.binary_mask)
            )
        # endregion compute additional information -----------------------------

        # sort the masks by size
        masks, shifts, sizes, bounding_boxes = zip(
            *sorted(zip(masks, shifts, sizes, bounding_boxes), key=lambda x: x[2], reverse=True)
        )

        # remove masks that are just one pixel wide
        del_idx = []
        for i, mask in enumerate(masks):
            if (
                bounding_boxes[i][0] >= bounding_boxes[i][2] - 1
                or bounding_boxes[i][1] >= bounding_boxes[i][3] - 1
            ):
                del_idx.append(i)

        masks = [mask for i, mask in enumerate(masks) if i not in del_idx]
        shifts = [shift for i, shift in enumerate(shifts) if i not in del_idx]
        sizes = [size for i, size in enumerate(sizes) if i not in del_idx]
        bounding_boxes = [bbox for i, bbox in enumerate(bounding_boxes) if i not in del_idx]

        # skip the first mask because it is often the background
        skips = config_parser.skip_idx[model_config["method"]]

        # check if there are enough masks
        if len(masks) <= skips:
            skips -= 1

        masks = masks[skips:]
        sizes = sizes[skips:]
        shifts = shifts[skips:]
        bounding_boxes = bounding_boxes[skips:]

        # region create mask dict ---------------------------------------------
        mask_dict = {
            "masks": [helpers.cut_image_by_single_mask(img, mask) for mask in masks],
            "size": sizes,
            "shift": shifts,
            "bounding_boxes": bounding_boxes,
        }
        # endregion create mask dict ------------------------------------------

        # save the segmentation masks
        dataset.save_image_segmentation_masks(model_config["method"], img_idx, mask_dict)

    dataset.save_segmentation_info()
