import os

import albumentations as A
import numpy as np
import skimage as ski
import torch
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import shift

import quanproto.dataloader.params as quan_params
import quanproto.datasets.config_parser as quan_dataset
from quanproto.augmentation import enums
from quanproto.datasets import functional as F
from quanproto.features.config_parser import feature_dict


class MultiLabelImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transform: A.Compose, info, crop) -> None:
        self.root = root
        self.paths = info["paths"]
        self.labels = info["labels"]
        self.class_ids = info["class_ids"]

        if "bboxes" in info and crop:
            self.bboxes = info["bboxes"]
            self.transform = A.Compose(
                enums.get_augmentation_pipeline("crop") + transform
            )
        else:
            self.transform = A.Compose(transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = ski.io.imread(os.path.join(self.root, self.paths[idx]))
        if len(img.shape) == 2:
            # convert to 3 channels
            img = ski.color.gray2rgb(img)
        if img.shape[2] == 4:
            img = (ski.color.rgba2rgb(img) * 255).astype(np.uint8)
        if img.shape[2] == 2:
            raise ValueError("Image has 2 channels")

        if hasattr(self, "bboxes"):
            largest_bbox = F.combine_bounding_boxes(self.bboxes[idx])
            img = self.transform(image=img, cropping_bbox=largest_bbox)["image"]
        else:
            img = self.transform(image=img)["image"]

        # check what dtype is returned
        if isinstance(img, torch.Tensor):
            img = img.float()
        else:
            img = torch.tensor(img).float()

        # make a tensor of labels
        labels = torch.tensor(self.labels[idx]).float()
        class_id = int(self.class_ids[idx][0])

        return img, labels, class_id


def test_dataloader(
    config,
    batch_size=quan_params.BATCH_SIZE,
    num_workers=quan_params.NUM_DATALOADER_WORKERS,
    pin_memory=quan_params.PIN_MEMORY,
    crop: bool = False,
):

    dataset = quan_dataset.get_dataset(config["dataset_dir"], config["dataset"])

    mean = feature_dict[config["features"]]["mean"]
    std = feature_dict[config["features"]]["std"]

    size = feature_dict[config["features"]]["input_size"][1:]

    test_dataset = MultiLabelImageDataset(
        dataset.test_dirs()["test"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        dataset.test_info(),
        crop,
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
    config,
    batch_size=quan_params.BATCH_SIZE,
    num_workers=quan_params.NUM_DATALOADER_WORKERS,
    pin_memory=quan_params.PIN_MEMORY,
    crop: bool = False,
):

    dataset = quan_dataset.get_dataset(config["dataset_dir"], config["dataset"])

    mean = feature_dict[config["features"]]["mean"]
    std = feature_dict[config["features"]]["std"]
    size = feature_dict[config["features"]]["input_size"][1:]

    fold_idx = config["fold_idx"]

    validation_dataset = MultiLabelImageDataset(
        dataset.fold_dirs(fold_idx)["validation"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        dataset.fold_info(fold_idx, "validation"),
        crop,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return validation_loader
