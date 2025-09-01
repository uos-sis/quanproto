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
from quanproto.datasets import functional as F
from quanproto.features.config_parser import feature_dict


class SingleAugmentationDataset(Dataset):
    def __init__(self, root: str, transform: list, info, crop: bool) -> None:
        self.root = root
        self.img_ids = info["ids"]  # important for the push
        self.paths = info["paths"]
        self.labels = info["labels"]
        self.multi_label = len(self.labels[0]) > 1

        if crop and "bboxes" in info:
            self.bboxes = info["bboxes"]
            self.transform = A.Compose(enums.get_augmentation_pipeline("crop") + transform)
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
            bbox = F.get_random_bounding_box(self.bboxes[idx])
            img = self.transform(image=img, cropping_bbox=bbox)["image"]
        else:
            img = self.transform(image=img)["image"]

        # check what dtype is returned
        if isinstance(img, torch.Tensor):
            img = img.float()
        else:
            img = torch.tensor(img).float()

        # make a tensor of labels
        if self.multi_label:
            labels = torch.tensor(self.labels[idx]).float()
        else:
            labels = int(self.labels[idx][0])
        return img, labels

    def getitem_by_id(self, img_id):
        """
        Get the item by id
        """
        # idx = self.img_ids.index(img_id)
        idx = img_id
        if idx is None:
            raise ValueError(f"Image id {img_id} not found in dataset")
        return self.__getitem__(idx)


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

    test_dataset = SingleAugmentationDataset(
        dataset.test_dirs()["test"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        dataset.test_info(),
        crop,
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

    validation_dataset = SingleAugmentationDataset(
        dataset.fold_dirs(fold_idx)["validation"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        dataset.fold_info(fold_idx, "validation"),
        crop,
    )
    validation_loader = DataLoader(
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
):
    """
    the prune dataloader is a train dataloader without augmentation and shuffle
    """

    dataset = quan_dataset.get_dataset(config["dataset_dir"], config["dataset"])

    mean = feature_dict[config["features"]]["mean"]
    std = feature_dict[config["features"]]["std"]
    size = feature_dict[config["features"]]["input_size"][1:]

    fold_idx = config["fold_idx"]

    train_dataset = SingleAugmentationDataset(
        dataset.fold_dirs(fold_idx)["train"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        dataset.fold_info(fold_idx, "train"),
        crop,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader


def make_dataloader_trainingset(
    config,
    num_workers=quan_params.NUM_DATALOADER_WORKERS,
    pin_memory=quan_params.PIN_MEMORY,
    crop: bool = False,
):
    dataset = quan_dataset.get_dataset(config["dataset_dir"], config["dataset"])

    mean = feature_dict[config["features"]]["mean"]
    std = feature_dict[config["features"]]["std"]

    size = feature_dict[config["features"]]["input_size"][1:]

    pipline = [A.Resize(size[0], size[1])]
    for key, range_id in config["augmentation_pipeline"]:
        pipline += enums.get_augmentation_pipeline(key, range_id)

    train_dataset = SingleAugmentationDataset(
        dataset.fold_dirs(config["fold_idx"])["train"],
        pipline
        + [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        dataset.fold_info(config["fold_idx"], "train"),
        crop,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if "validation" in dataset.fold_dirs(config["fold_idx"]):
        validation_dataset = SingleAugmentationDataset(
            dataset.fold_dirs(config["fold_idx"])["validation"],
            [
                A.Resize(size[0], size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            dataset.fold_info(config["fold_idx"], "validation"),
            crop,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        # use the test folder as validation
        validation_dataset = SingleAugmentationDataset(
            dataset.test_dirs()["test"],
            [
                A.Resize(size[0], size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            dataset.test_info(),
            crop,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    push_dataset = SingleAugmentationDataset(
        dataset.fold_dirs(config["fold_idx"])["train"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        dataset.fold_info(config["fold_idx"], "train"),
        crop,
    )
    push_loader = DataLoader(
        push_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    class_weights = dataset.class_weights(config["fold_idx"])

    return train_loader, validation_loader, push_loader, class_weights
