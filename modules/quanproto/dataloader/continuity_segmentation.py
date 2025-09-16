import os

import albumentations as A
import numpy as np
import skimage as ski
import torch
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import shift
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


class ContinuitySegmentationMaskDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform1: list,
        transform2: list,
        explain_transform: list,
        info,
        mask_method,
        num_masks: int = 0,
        crop: bool = True,
        fill_background_method: str = "zero",
        min_bbox_size: float = 0.0,
    ) -> None:
        self.root = os.path.join(root, mask_method)
        self.paths = info["masks"][mask_method]["paths"]
        self.img_ids = info["ids"]  # important for the push
        self.labels = info["labels"]
        self.fill_background_method = fill_background_method

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

        self.explain_transform = A.Compose(explain_transform)

        self.max_partlocs = 0
        if "partlocs" in info:
            self.partlocs = info["partlocs"]
            # get the maximum number of keys in a partlocs dict
            for partloc in self.partlocs:
                self.max_partlocs = max(self.max_partlocs, len(partloc.keys()))

        if crop and "bounding_boxes" in info["masks"][mask_method]:
            self.bboxes = info["masks"][mask_method]["bounding_boxes"]
            transform1 = enums.get_augmentation_pipeline("crop") + transform1
            transform2 = enums.get_augmentation_pipeline("crop") + transform2

        if hasattr(self, "partlocs"):
            self.trans_pip1 = A.Compose(
                transform1,
                keypoint_params=A.KeypointParams(format="xy"),
            )
            self.trans_pip2 = A.Compose(
                transform2,
                keypoint_params=A.KeypointParams(format="xy"),
            )
        else:
            self.trans_pip1 = A.Compose(transform1)
            self.trans_pip2 = A.Compose(transform2)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        # start with an empty tensor
        tf1_masks = torch.empty(0)
        tf2_masks = torch.empty(0)
        uncropped_masks = torch.empty(0)
        # make a tensor of labels
        if self.multi_label:
            label = torch.tensor(self.labels[idx]).float()
        else:
            label = torch.tensor(self.labels[idx][0]).long()

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

            width, height = img.shape[1], img.shape[0]

        for i in range(num_masks):
            mask_path = self.paths[idx][i]
            original_mask = ski.io.imread(os.path.join(self.root, mask_path))

            if len(original_mask.shape) == 2:
                original_mask = ski.color.gray2rgb(original_mask)
            if original_mask.shape[2] == 4:
                original_mask = (ski.color.rgba2rgb(original_mask) * 255).astype(np.uint8)
            if original_mask.shape[2] == 2:
                original_mask = ski.color.gray2rgb(original_mask[:, :, 0])

            input_mask = original_mask.copy()
            if self.fill_background_method == "mean":
                input_mask = add_mean_background(input_mask)

            if self.fill_background_method == "original":
                input_mask = add_original_background(input_mask, img)

            if hasattr(self, "partlocs") and hasattr(self, "bboxes"):
                partlocs = list(self.partlocs[idx].values())

                bboxes = self.bboxes[idx][i]
                if self.min_bbox_size > 0:
                    bboxes = expand_bboxes(bboxes, (width, height), self.min_bbox_size)[0]

                # first transform the keypoints based on the image
                t = self.trans_pip1(
                    image=input_mask,
                    keypoints=partlocs,
                    cropping_bbox=bboxes,
                )
                tf_partlocs = t["keypoints"]
                tf1_mask = t["image"]

                t = self.trans_pip2(
                    image=input_mask,
                    keypoints=partlocs,
                    cropping_bbox=bboxes,
                )
                tf2_mask = t["image"]

            if hasattr(self, "partlocs") and not hasattr(self, "bboxes"):
                # first transform the keypoints based on the image
                partlocs = list(self.partlocs[idx].values())
                t = self.trans_pip1(image=input_mask, keypoints=partlocs)
                tf_partlocs = t["keypoints"]
                tf1_mask = t["image"]

                t = self.trans_pip2(image=input_mask, keypoints=partlocs)
                tf2_mask = t["image"]

            if not hasattr(self, "partlocs") and hasattr(self, "bboxes"):
                bboxes = self.bboxes[idx][i]
                if self.min_bbox_size > 0:
                    bboxes = expand_bboxes(bboxes, (width, height), self.min_bbox_size)[0]
                # first transform the keypoints based on the image
                t = self.trans_pip1(image=input_mask, cropping_bbox=bboxes)
                tf1_mask = t["image"]

                t = self.trans_pip2(image=input_mask, cropping_bbox=bboxes)
                tf2_mask = t["image"]

            if not hasattr(self, "partlocs") and not hasattr(self, "bboxes"):
                t = self.trans_pip1(image=input_mask)
                tf1_mask = t["image"]

                t = self.trans_pip2(image=input_mask)
                tf2_mask = t["image"]

            uncropped_mask = self.explain_transform(image=original_mask)["image"]

            # check what dtype is returned
            if isinstance(tf1_mask, torch.Tensor):
                tf1_mask = tf1_mask.float()
            else:
                tf1_mask = torch.tensor(tf1_mask).float()

            if isinstance(tf2_mask, torch.Tensor):
                tf2_mask = tf2_mask.float()
            else:
                tf2_mask = torch.tensor(tf2_mask).float()

            if isinstance(uncropped_mask, torch.Tensor):
                uncropped_mask = uncropped_mask.float()
            else:
                uncropped_mask = torch.tensor(uncropped_mask).float()

            tf1_masks = torch.cat((tf1_masks, tf1_mask.unsqueeze(0)), 0)
            tf2_masks = torch.cat((tf2_masks, tf1_mask.unsqueeze(0)), 0)
            uncropped_masks = torch.cat((uncropped_masks, uncropped_mask.unsqueeze(0)), 0)

        # make a tensor with (max_partlocs, 2) and fill it with -1
        if self.max_partlocs > 0:
            partlocs = torch.full((self.max_partlocs, 2), -1, dtype=torch.float32)
        else:
            partlocs = torch.tensor([])

        if hasattr(self, "partlocs"):
            # overwrite the -1 entries with the actual partlocs
            for i, point in enumerate(tf_partlocs):
                partlocs[i] = torch.tensor([point[0], point[1]], dtype=torch.float32)

        return (tf1_masks, uncropped_masks), tf2_masks, partlocs, label


def test_dataloader(
    config: dict,
    batch_size=quan_params.BATCH_SIZE,
    num_workers=quan_params.NUM_DATALOADER_WORKERS,
    pin_memory=quan_params.PIN_MEMORY,
    crop: bool = True,
    fill_background_method: str = "zero",
    min_bbox_size: float = 0.0,
):
    dataset = quan_dataset.get_dataset(config["dataset_dir"], config["dataset"])

    mean = feature_dict[config["features"]]["mean"]
    std = feature_dict[config["features"]]["std"]
    size = feature_dict[config["features"]]["input_size"][1:]

    segmentation_method = config["segmentation_method"]

    aug_pipeline_2 = [A.Resize(size[0], size[1])]
    aug_pipeline_2 += enums.get_augmentation_pipeline("continuity")
    aug_pipeline_2 += [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    test_dataset = ContinuitySegmentationMaskDataset(
        dataset.test_dirs()["segmentations"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        aug_pipeline_2,
        [
            A.Resize(size[0], size[1]),
            ToTensorV2(),
        ],
        dataset.test_info(),
        segmentation_method,
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
    crop: bool = True,
    fill_background_method: str = "zero",
    min_bbox_size: float = 0.0,
):
    dataset = quan_dataset.get_dataset(config["dataset_dir"], config["dataset"])

    mean = feature_dict[config["features"]]["mean"]
    std = feature_dict[config["features"]]["std"]
    size = feature_dict[config["features"]]["input_size"][1:]

    segmentation_method = config["segmentation_method"]
    fold_idx = config["fold_idx"]

    aug_pipeline_2 = [A.Resize(size[0], size[1])]
    aug_pipeline_2 += enums.get_augmentation_pipeline("continuity")
    aug_pipeline_2 += [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    validation_dataset = ContinuitySegmentationMaskDataset(
        dataset.fold_dirs(fold_idx)["validation_segmentations"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        aug_pipeline_2,
        [
            A.Resize(size[0], size[1]),
            ToTensorV2(),
        ],
        dataset.fold_info(fold_idx, "validation"),
        segmentation_method,
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
