import os

import albumentations as A
import numpy as np
import skimage as ski
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

import quanproto.dataloader.params as quan_params
import quanproto.datasets.config_parser as quan_dataset
from quanproto.augmentation import enums
from quanproto.dataloader.helpers import (
    add_mean_background,
    add_original_background,
    expand_bboxes,
)
from quanproto.datasets import functional as F
from quanproto.features.config_parser import feature_dict


class ComplexitySegmentationMaskDataset(Dataset):
    def __init__(
        self,
        root,
        transform: list,
        explain_transform: list,
        info,
        mask_method,
        num_masks: int = 0,
        crop: bool = False,
        fill_background_method: str = "zero",
        min_bbox_size: float = 0.0,
    ) -> None:
        self.root = os.path.join(root, mask_method)
        self.paths = info["masks"][mask_method]["paths"]
        self.img_ids = info["ids"]  # important for the push
        self.labels = info["labels"]
        self.fill_background_method = fill_background_method

        self.seg_mask_root = os.path.join(root, "original")
        self.seg_masks = info["masks"]["original"]["paths"]

        self.min_bbox_size = min_bbox_size

        self.img_root = root
        last_part = self.img_root.split("/")[-1]

        # count number of _ in the last part
        num_ = last_part.count("_")

        if num_ > 0:
            self.img_root = self.img_root.replace("_segmentations", "")
        else:
            self.img_root = "/".join(self.img_root.split("/")[:-1])
            # add test as last part
            self.img_root = os.path.join(self.img_root, "test")

        self.multi_label = len(self.labels[0]) > 1

        self.num_masks = num_masks

        if self.fill_background_method == "original":
            self.img_paths = info["paths"]

        self.max_partlocs = 0
        if "partlocs" in info:
            self.partlocs = info["partlocs"]
            # get the maximum number of keys in a partlocs dict
            for partloc in self.partlocs:
                self.max_partlocs = max(self.max_partlocs, len(partloc.keys()))

        if crop and "bounding_boxes" in info["masks"][mask_method]:
            self.bboxes = info["masks"][mask_method]["bounding_boxes"]
            transform = enums.get_augmentation_pipeline("crop") + transform
        self.trans_pip = A.Compose(transform)

        if hasattr(self, "partlocs"):
            self.explain_transform = A.Compose(
                explain_transform,
                keypoint_params=A.KeypointParams(format="xy"),
            )
        else:
            self.explain_transform = A.Compose(explain_transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        # make a tensor of labels
        if self.multi_label:
            label = torch.tensor(self.labels[idx]).float()
        else:
            label = torch.tensor(self.labels[idx][0]).long()

        original_bboxes = []
        original_mask_size = []

        seg_mask = ski.io.imread(os.path.join(self.seg_mask_root, self.seg_masks[idx][0]))
        if len(seg_mask.shape) == 2:
            # convert to 3 channels
            seg_mask = ski.color.gray2rgb(seg_mask)
        if seg_mask.shape[2] == 4:
            seg_mask = (ski.color.rgba2rgb(seg_mask) * 255).astype(np.uint8)
        if seg_mask.shape[2] == 2:
            seg_mask = ski.color.gray2rgb(seg_mask[:, :, 0])

        original_mask_size = seg_mask.shape[:2]

        if hasattr(self, "partlocs"):
            partlocs = list(self.partlocs[idx].values())
            partlocs_ids = list(self.partlocs[idx].keys())
            # transform the segmentation mask based on the image
            t = self.explain_transform(
                image=seg_mask,
                keypoints=partlocs,
            )
            seg_mask = t["image"]
            tf_partlocs = t["keypoints"]
        else:
            seg_mask = self.explain_transform(image=seg_mask)["image"]

        # check what dtype is returned
        if isinstance(seg_mask, torch.Tensor):
            seg_mask = seg_mask.float()
        else:
            seg_mask = torch.tensor(seg_mask).float()

        tf_masks = torch.empty(0)
        uncropped_masks = torch.empty(0)

        if self.num_masks > 0:
            num_masks = self.num_masks
        else:
            num_masks = len(self.paths[idx])

        # load the image just once
        if self.fill_background_method == "original":

            img = ski.io.imread(os.path.join(self.img_root, self.img_paths[idx]))
            if len(img.shape) == 2:
                img = ski.color.gray2rgb(img)
            if img.shape[2] == 4:
                img = (ski.color.rgba2rgb(img) * 255).astype(np.uint8)
            if img.shape[2] == 2:
                img = ski.color.gray2rgb(img[:, :, 0])

        for i in range(num_masks):
            mask_path = self.paths[idx][i]
            original_mask = ski.io.imread(os.path.join(self.root, mask_path))

            if len(original_mask.shape) == 2:
                original_mask = ski.color.gray2rgb(original_mask)
            if original_mask.shape[2] == 4:
                original_mask = (ski.color.rgba2rgb(original_mask) * 255).astype(np.uint8)
            if original_mask.shape[2] == 2:
                original_mask = ski.color.gray2rgb(original_mask[:, :, 0])

            width, height = original_mask.shape[1], original_mask.shape[0]

            input_mask = original_mask.copy()
            if self.fill_background_method == "mean":
                input_mask = add_mean_background(input_mask)

            if self.fill_background_method == "original":
                input_mask = add_original_background(input_mask, img)

            if hasattr(self, "bboxes"):
                bboxes = self.bboxes[idx][i]
                if self.min_bbox_size > 0:
                    bboxes = expand_bboxes(bboxes, (width, height), self.min_bbox_size)[0]

                original_bboxes.append(bboxes)
                # first transform the keypoints based on the image
                t = self.trans_pip(
                    image=input_mask,
                    cropping_bbox=bboxes,
                )
                tf_mask = t["image"]

            if hasattr(self, "partlocs"):
                uncropped_mask = self.explain_transform(image=original_mask, keypoints=partlocs)[
                    "image"
                ]
            else:
                uncropped_mask = self.explain_transform(image=original_mask)["image"]

            # check what dtype is returned
            if isinstance(tf_mask, torch.Tensor):
                tf_mask = tf_mask.float()
            else:
                tf_mask = torch.tensor(tf_mask).float()

            if isinstance(uncropped_mask, torch.Tensor):
                uncropped_mask = uncropped_mask.float()
            else:
                uncropped_mask = torch.tensor(uncropped_mask).float()

            tf_masks = torch.cat((tf_masks, tf_mask.unsqueeze(0)), 0)

            uncropped_masks = torch.cat((uncropped_masks, uncropped_mask.unsqueeze(0)), 0)

        # make a tensor with (max_partlocs, 2) and fill it with -1
        if self.max_partlocs > 0:
            t_partlocs = torch.full((self.max_partlocs, 2), -1, dtype=torch.float32)
            t_partlocs_ids = torch.full((self.max_partlocs, 1), -1, dtype=torch.int32)
        else:
            t_partlocs = torch.tensor([])
            t_partlocs_ids = torch.tensor([])

        if hasattr(self, "partlocs"):
            # overwrite the -1 entries with the actual partlocs
            for i, point in enumerate(tf_partlocs):
                # switch the value from x,y to y,x
                t_partlocs[i] = torch.tensor([point[0], point[1]], dtype=torch.float32)
                t_partlocs_ids[i] = torch.tensor(partlocs_ids[i], dtype=torch.int32)

        # make a tensor from the bboxes
        original_bboxes = torch.tensor(original_bboxes, dtype=torch.float32)
        original_mask_size = torch.tensor(original_mask_size, dtype=torch.float32)

        return (
            (tf_masks, uncropped_masks, original_bboxes, original_mask_size),
            seg_mask,
            t_partlocs,
            t_partlocs_ids,
            label,
        )


def test_dataloader(
    config: dict,
    batch_size=quan_params.BATCH_SIZE,
    num_workers=quan_params.NUM_DATALOADER_WORKERS,
    pin_memory=quan_params.PIN_MEMORY,
    crop: bool = False,
    fill_background_method: str = "zero",
    min_bbox_size: float = 0.0,
):
    dataset = quan_dataset.get_dataset(config["dataset_dir"], config["dataset"])

    mean = feature_dict[config["features"]]["mean"]
    std = feature_dict[config["features"]]["std"]
    size = feature_dict[config["features"]]["input_size"][1:]

    test_dataset = ComplexitySegmentationMaskDataset(
        dataset.test_dirs()["segmentations"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        [
            A.Resize(size[0], size[1]),
            ToTensorV2(),
        ],
        dataset.test_info(),
        config["segmentation_method"],
        config["num_masks"],
        crop,
        fill_background_method,
        min_bbox_size=min_bbox_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_loader


def validation_dataloader(
    config: dict,
    batch_size=quan_params.BATCH_SIZE,
    num_workers=quan_params.NUM_DATALOADER_WORKERS,
    pin_memory=quan_params.PIN_MEMORY,
    crop: bool = False,
    fill_background_method: str = "zero",
    min_bbox_size: float = 0.0,
):
    dataset = quan_dataset.get_dataset(config["dataset_dir"], config["dataset"])

    mean = feature_dict[config["features"]]["mean"]
    std = feature_dict[config["features"]]["std"]
    size = feature_dict[config["features"]]["input_size"][1:]

    fold_idx = config["fold_idx"]

    assert "validation" in dataset.fold_dirs(fold_idx), "Validation set not found"

    validation_dataset = ComplexitySegmentationMaskDataset(
        dataset.fold_dirs(fold_idx)["validation_segmentations"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        [
            A.Resize(size[0], size[1]),
            ToTensorV2(),
        ],
        dataset.fold_info(fold_idx, "validation"),
        config["segmentation_method"],
        config["num_masks"],
        crop,
        fill_background_method,
        min_bbox_size=min_bbox_size,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return validation_loader
