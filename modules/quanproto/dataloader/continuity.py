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


class ContinuityDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform1: A.Compose,
        transform2: A.Compose,
        info,
        crop: bool = False,
    ) -> None:
        self.root = root
        self.paths = info["paths"]
        self.labels = info["labels"]
        self.crop = crop
        self.multi_label = len(self.labels[0]) > 1

        self.max_partlocs = 0
        if "partlocs" in info:
            self.partlocs = info["partlocs"]
            # get the maximum number of keys in a partlocs dict
            for partloc in self.partlocs:
                self.max_partlocs = max(self.max_partlocs, len(partloc.keys()))

        self.trans_pip1 = []
        self.trans_pip2 = []
        if "bboxes" in info and self.crop:
            self.bboxes = info["bboxes"]
            self.trans_pip1 += enums.get_augmentation_pipeline("crop") + transform1
            self.trans_pip2 += enums.get_augmentation_pipeline("crop") + transform2
        else:
            self.trans_pip1 += transform1
            self.trans_pip2 += transform2

        if hasattr(self, "partlocs"):
            self.trans_pip1 = A.Compose(
                self.trans_pip1,
                keypoint_params=A.KeypointParams(format="xy"),
            )
            self.trans_pip2 = A.Compose(
                self.trans_pip2,
                keypoint_params=A.KeypointParams(format="xy"),
            )
        else:
            self.trans_pip1 = A.Compose(
                self.trans_pip1,
            )
            self.trans_pip2 = A.Compose(
                self.trans_pip2,
            )

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

        # just crop
        if hasattr(self, "bboxes") and not hasattr(self, "partlocs"):
            largest_bbox = F.combine_bounding_boxes(self.bboxes[idx])
            img_t1 = self.trans_pip1(image=img, cropping_bbox=largest_bbox)["image"]
            img_t2 = self.trans_pip2(image=img, cropping_bbox=largest_bbox)["image"]

        # crop and transform keypoints
        if hasattr(self, "bboxes") and hasattr(self, "partlocs"):
            largest_bbox = F.combine_bounding_boxes(self.bboxes[idx])

            partlocs = [p for p in self.partlocs[idx].values()]
            t1 = self.trans_pip1(image=img, cropping_bbox=largest_bbox, keypoints=partlocs)
            tf_partlocs = t1["keypoints"]
            img_t1 = t1["image"]

            t2 = self.trans_pip2(image=img, cropping_bbox=largest_bbox, keypoints=partlocs)
            img_t2 = t2["image"]
            # INFO t2 has only additional photometric augmentations so the keypoints are the same

        # just transform keypoints
        if not hasattr(self, "bboxes") and hasattr(self, "partlocs"):
            partlocs = [p for p in self.partlocs[idx].values()]
            t1 = self.trans_pip1(image=img, keypoints=partlocs)
            tf_partlocs = t1["keypoints"]
            img_t1 = t1["image"]

            t2 = self.trans_pip2(image=img, keypoints=partlocs)
            img_t2 = t2["image"]
            # INFO t2 has only additional photometric augmentations so the keypoints are the same

        # no cropping or keypoints
        if not hasattr(self, "bboxes") and not hasattr(self, "partlocs"):
            img_t1 = self.trans_pip1(image=img)["image"]
            img_t2 = self.trans_pip2(image=img)["image"]

        # check what dtype is returned
        if isinstance(img_t1, torch.Tensor) and isinstance(img_t2, torch.Tensor):
            img_t1 = img_t1.float()
            img_t2 = img_t2.float()
        else:
            img_t1 = torch.tensor(img_t1).float()
            img_t2 = torch.tensor(img_t2).float()

        # make a tensor of labels
        if self.multi_label:
            labels = torch.tensor(self.labels[idx]).float()
        else:
            labels = int(self.labels[idx][0])

        # make a tensor with (max_partlocs, 2) and fill it with -1
        if self.max_partlocs > 0:
            partlocs = torch.full((self.max_partlocs, 2), -1, dtype=torch.float32)
        else:
            partlocs = torch.tensor([])
        if hasattr(self, "partlocs"):
            # overwrite the -1 entries with the actual partlocs
            for i, point in enumerate(tf_partlocs):
                partlocs[i] = torch.tensor([point[0], point[1]], dtype=torch.float32)

        return img_t1, img_t2, partlocs, labels


def test_dataloader(
    config: dict,
    batch_size=quan_params.BATCH_SIZE,
    num_workers=quan_params.NUM_DATALOADER_WORKERS,
    pin_memory=quan_params.PIN_MEMORY,
    crop: bool = False,
):
    dataset = quan_dataset.get_dataset(config["dataset_dir"], config["dataset"])

    mean = feature_dict[config["features"]]["mean"]
    std = feature_dict[config["features"]]["std"]
    size = feature_dict[config["features"]]["input_size"][1:]

    aug_pipeline_2 = [A.Resize(size[0], size[1])]
    aug_pipeline_2 += enums.get_augmentation_pipeline("continuity")
    aug_pipeline_2 += [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    test_dataset = ContinuityDataset(
        dataset.test_dirs()["test"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        aug_pipeline_2,
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
    config: dict,
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

    aug_pipeline_2 = [A.Resize(size[0], size[1])]
    aug_pipeline_2 += enums.get_augmentation_pipeline("continuity")
    aug_pipeline_2 += [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    validation_dataset = ContinuityDataset(
        dataset.fold_dirs(fold_idx)["validation"],
        [
            A.Resize(size[0], size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        aug_pipeline_2,
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
