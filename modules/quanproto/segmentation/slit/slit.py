import math
from itertools import product

import cv2
import numpy as np
import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

import quanproto.segmentation.best_params as best_params

# This code was taken from https://github.com/ChenDelong1999/subobjects/tree/main
# new functionality to create the models was added
# You can find the reference here:
# @article{chen2024subobject,
#  author       = {Delong Chen and
#                  Samuel Cahyawijaya and
#                  Jianfeng Liu and
#                  Baoyuan Wang and
#                  Pascale Fung},
#  title        = {Subobject-level Image Tokenization},
#  journal      = {CoRR},
#  volume       = {abs/2402.14327},
#  year         = {2024},
#  url          = {https://doi.org/10.48550/arXiv.2402.14327},
#  doi          = {10.48550/ARXIV.2402.14327},
#  eprinttype    = {arXiv},
#  eprint       = {2402.14327}
# }


def generate_crop_boxes(im_size, n_layers=1, overlap=0):
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """

    crop_boxes, layer_idxs = [], []
    im_w, im_h = im_size

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        # overlap = int(overlap_ratio * min(im_h, im_w) * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    # only keep layer_id=n_layers
    crop_boxes = [box for box, layer in zip(crop_boxes, layer_idxs) if layer == n_layers]
    layer_idxs = [layer for layer in layer_idxs if layer == n_layers]

    return crop_boxes


def inference_single_image(
    image, image_processor, model, pyramid_layers=0, overlap=90, resolution=None
):
    if resolution:
        image_processor.size["height"] = resolution
        image_processor.size["width"] = resolution

    def run(image, bzp=0):
        encoding = image_processor(image, return_tensors="pt")
        pixel_values = encoding.pixel_values.to(model.device).to(model.dtype)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.float().cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=(image.shape[0], image.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        probs = torch.sigmoid(upsampled_logits)[0, 0].detach().numpy()

        if bzp > 0:
            probs = boundary_zero_padding(probs, p=bzp)
        return probs

    global_probs = run(image)

    if pyramid_layers > 0:
        for layer in range(1, pyramid_layers + 1):
            boxes = generate_crop_boxes(image.size, n_layers=layer, overlap=overlap)
            for box in boxes:
                x1, y1, x2, y2 = box
                crop = image.crop(box)
                probs = run(crop, bzp=overlap)
                global_probs[y1:y2, x1:x2] += probs
        global_probs /= pyramid_layers + 1

    return global_probs


def probs_to_masks(probs, threshold=0.0025):

    # delate the probs to close some gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.dilate((probs > threshold).astype(np.uint8), kernel)

    binarilized = (result < threshold).astype(np.uint8)
    num_objects, labels = cv2.connectedComponents(binarilized)
    masks = [labels == i for i in range(1, labels.max() + 1)]
    masks.sort(key=lambda x: x.sum(), reverse=True)
    return masks


def boundary_zero_padding(probs, p=15):
    # from https://arxiv.org/abs/2308.13779

    zero_p = p // 3
    alpha_p = zero_p * 2

    probs[:, :alpha_p] *= 0.5
    probs[:, -alpha_p:] *= 0.5
    probs[:alpha_p, :] *= 0.5
    probs[-alpha_p:, :] *= 0.5

    probs[:, :zero_p] = 0
    probs[:, -zero_p:] = 0
    probs[:zero_p, :] = 0
    probs[-zero_p:, :] = 0

    return probs


def create_slit_model(model_config):
    checkpoint = model_config["checkpoint"]
    model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint).to("cuda").eval()
    preprocessor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)
    return preprocessor, model, model_config["threshold"]


def segment_image(model, img, bbox):
    probs_list = inference_single_image(img, model[0], model[1])
    mask_list = probs_to_masks(probs_list, model[2])
    return mask_list
