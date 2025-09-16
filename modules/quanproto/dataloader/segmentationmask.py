import os
import sys

import albumentations as A
import numpy as np
import skimage as ski
import torch.utils.data
from albumentations.pytorch import ToTensorV2

import quanproto.dataloader.params as quan_params
import quanproto.datasets.config_parser as quan_dataset
from quanproto.augmentation import enums
from quanproto.dataloader.helpers import (
    add_mean_background,
    add_original_background,
    expand_bboxes,
)
from quanproto.features.config_parser import feature_dict
from quanproto.utils.vis_helper import save_image


class SegmentaionMaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        transform: list,
        info: dict,
        mask_method: str,
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

        if crop and "bounding_boxes" in info["masks"][mask_method]:
            self.bboxes = info["masks"][mask_method]["bounding_boxes"]
            self.transform = A.Compose(enums.get_augmentation_pipeline("crop") + transform)
        else:
            self.transform = A.Compose(transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        # start with an empty tensor
        masks = torch.empty(0)
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
            mask = ski.io.imread(os.path.join(self.root, mask_path))

            if len(mask.shape) == 2:
                mask = ski.color.gray2rgb(mask)
            if mask.shape[2] == 4:
                mask = (ski.color.rgba2rgb(mask) * 255).astype(np.uint8)
            if mask.shape[2] == 2:
                mask = ski.color.gray2rgb(mask[:, :, 0])

            if self.fill_background_method == "mean":
                mask = add_mean_background(mask)

            if self.fill_background_method == "original":
                mask = add_original_background(mask, img)

            if hasattr(self, "bboxes"):
                bboxes = self.bboxes[idx][i]
                if self.min_bbox_size > 0:
                    bboxes = expand_bboxes(bboxes, (width, height), self.min_bbox_size)[0]

                mask = self.transform(image=mask, cropping_bbox=bboxes)["image"]
            else:
                mask = self.transform(image=mask)["image"]

            # check what dtype is returned
            if isinstance(mask, torch.Tensor):
                mask = mask.float()
            else:
                mask = torch.tensor(mask).float()

            masks = torch.cat((masks, mask.unsqueeze(0)), 0)

        return masks, label


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

    test_dataset = SegmentaionMaskDataset(
        dataset.test_dirs()["segmentations"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        dataset.test_info(),
        segmentation_method,
        config["num_masks"],
        crop,
        fill_background_method,
        min_bbox_size=min_bbox_size,
    )
    test_loader = torch.utils.data.DataLoader(
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

    fold_idx = config["fold_idx"]
    segmentation_method = config["segmentation_method"]

    validation_dataset = SegmentaionMaskDataset(
        dataset.fold_dirs(fold_idx)["validation_segmentations"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        dataset.fold_info(fold_idx, "validation"),
        segmentation_method,
        config["num_masks"],
        crop,
        fill_background_method,
        min_bbox_size=min_bbox_size,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return validation_loader


def prune_dataloader(
    config,
    batch_size=quan_params.BATCH_SIZE,
    num_workers=quan_params.NUM_DATALOADER_WORKERS,
    pin_memory=quan_params.PIN_MEMORY,
    crop: bool = False,
    fill_background_method: str = "zero",
    min_bbox_size: float = 0.0,
):
    """
    the prune dataloader is a train dataloader without augmentation and shuffle
    """
    dataset = quan_dataset.get_dataset(config["dataset_dir"], config["dataset"])

    mean = feature_dict[config["features"]]["mean"]
    std = feature_dict[config["features"]]["std"]
    size = feature_dict[config["features"]]["input_size"][1:]

    push_dataset = SegmentaionMaskDataset(
        dataset.fold_dirs(config["fold_idx"])["train_segmentations"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        dataset.fold_info(config["fold_idx"], "train"),
        config["segmentation_method"],
        config["num_masks"],
        crop,
        fill_background_method,
        min_bbox_size=min_bbox_size,
    )
    push_loader = torch.utils.data.DataLoader(
        push_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return push_loader


def make_dataloader_trainingset(
    config,
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

    pipline = [A.Resize(size[0], size[1])]
    for key, range_id in config["augmentation_pipeline"]:
        pipline += enums.get_augmentation_pipeline(key, range_id)

    train_dataset = SegmentaionMaskDataset(
        dataset.fold_dirs(config["fold_idx"])["train_segmentations"],
        pipline
        + [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        dataset.fold_info(config["fold_idx"], "train"),
        config["segmentation_method"],
        config["num_masks"],
        crop,
        fill_background_method,
        min_bbox_size=min_bbox_size,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if "validation" in dataset.fold_dirs(config["fold_idx"]):
        validation_dataset = SegmentaionMaskDataset(
            dataset.fold_dirs(config["fold_idx"])["validation_segmentations"],
            [
                A.Resize(size[0], size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            dataset.fold_info(config["fold_idx"], "validation"),
            config["segmentation_method"],
            config["num_masks"],
            crop,
            fill_background_method,
            min_bbox_size=min_bbox_size,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        # use the test folder as validation
        validation_dataset = SegmentaionMaskDataset(
            dataset.test_dirs()["segmentations"],
            [
                A.Resize(size[0], size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            dataset.test_info(),
            config["segmentation_method"],
            config["num_masks"],
            crop,
            fill_background_method,
            min_bbox_size=min_bbox_size,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    push_dataset = SegmentaionMaskDataset(
        dataset.fold_dirs(config["fold_idx"])["train_segmentations"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        dataset.fold_info(config["fold_idx"], "train"),
        config["segmentation_method"],
        config["num_masks"],
        crop,
        fill_background_method,
        min_bbox_size=min_bbox_size,
    )
    push_loader = torch.utils.data.DataLoader(
        push_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    class_weights = dataset.class_weights(config["fold_idx"])

    return train_loader, validation_loader, push_loader, class_weights
