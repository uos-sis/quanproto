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


class ComplexityDataset(Dataset):
    def __init__(
        self, image_root: str, mask_root: str, transform: A.Compose, info, crop=False
    ) -> None:
        self.image_root = image_root
        self.image_paths = info["paths"]

        self.seg_mask_root = os.path.join(mask_root, "original")
        self.seg_mask_paths = info["masks"]["original"]["paths"]

        self.labels = info["labels"]
        self.multi_label = len(self.labels[0]) > 1
        self.crop = crop

        self.max_partlocs = 0
        if "partlocs" in info:
            self.partlocs = info["partlocs"]
            # get the maximum number of keys in a partlocs dict
            for partloc in self.partlocs:
                self.max_partlocs = max(self.max_partlocs, len(partloc.keys()))

        self.transform = []
        if "bboxes" in info and self.crop:
            self.bboxes = info["bboxes"]
            self.transform += enums.get_augmentation_pipeline("crop") + transform
        else:
            self.transform += transform

        if hasattr(self, "partlocs"):
            self.transform = A.Compose(
                self.transform,
                keypoint_params=A.KeypointParams(format="xy"),
            )
        else:
            self.transform = A.Compose(
                self.transform,
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        seg_mask = ski.io.imread(
            os.path.join(self.seg_mask_root, self.seg_mask_paths[idx][0])
        )
        if len(seg_mask.shape) == 2:
            # convert to 3 channels
            seg_mask = ski.color.gray2rgb(seg_mask)
        if seg_mask.shape[2] == 4:
            seg_mask = (ski.color.rgba2rgb(seg_mask) * 255).astype(np.uint8)
        if seg_mask.shape[2] == 2:
            seg_mask = ski.color.gray2rgb(seg_mask[:, :, 0])

        img = ski.io.imread(os.path.join(self.image_root, self.image_paths[idx]))
        if len(img.shape) == 2:
            # convert to 3 channels
            img = ski.color.gray2rgb(img)
        if img.shape[2] == 4:
            img = (ski.color.rgba2rgb(img) * 255).astype(np.uint8)
        if img.shape[2] == 2:
            img = ski.color.gray2rgb(img[:, :, 0])

        # just crop
        if hasattr(self, "bboxes") and not hasattr(self, "partlocs"):
            largest_bbox = F.combine_bounding_boxes(self.bboxes[idx])

            t = self.transform(image=img, cropping_bbox=largest_bbox, mask=seg_mask)
            tf_img = t["image"]
            tf_seg_mask = t["mask"]

        # crop and transform keypoints
        if hasattr(self, "bboxes") and hasattr(self, "partlocs"):
            largest_bbox = F.combine_bounding_boxes(self.bboxes[idx])

            partlocs = [p for p in self.partlocs[idx].values()]
            partlocs_ids = [p for p in self.partlocs[idx].keys()]

            t = self.transform(
                image=img, cropping_bbox=largest_bbox, mask=seg_mask, keypoints=partlocs
            )
            tf_partlocs = t["keypoints"]
            tf_img = t["image"]
            tf_seg_mask = t["mask"]

        # just transform keypoints
        if not hasattr(self, "bboxes") and hasattr(self, "partlocs"):
            partlocs = [p for p in self.partlocs[idx].values()]
            partlocs_ids = [p for p in self.partlocs[idx].keys()]

            t = self.transform(image=img, mask=seg_mask, keypoints=partlocs)
            tf_partlocs = t["keypoints"]
            tf_img = t["image"]
            tf_seg_mask = t["mask"]

        if not hasattr(self, "bboxes") and not hasattr(self, "partlocs"):
            t = self.transform(image=img, mask=seg_mask)
            tf_img = t["image"]
            tf_seg_mask = t["mask"]

        # check what dtype is returned
        if isinstance(tf_img, torch.Tensor):
            tf_img = tf_img.float()
        else:
            tf_img = torch.tensor(tf_img).float()

        if isinstance(tf_seg_mask, torch.Tensor):
            tf_seg_mask = tf_seg_mask.float()
        else:
            tf_seg_mask = torch.tensor(tf_seg_mask).float()

        # make a tensor of labels
        if self.multi_label:
            labels = torch.tensor(self.labels[idx]).float()
        else:
            labels = torch.tensor(self.labels[idx][0]).long()

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
                t_partlocs[i] = torch.tensor([point[1], point[0]], dtype=torch.float32)
                t_partlocs_ids[i] = torch.tensor(partlocs_ids[i], dtype=torch.int32)

        return tf_img, tf_seg_mask, t_partlocs, t_partlocs_ids, labels


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

    test_dataset = ComplexityDataset(
        dataset.test_dirs()["test"],
        dataset.test_dirs()["segmentations"],
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

    assert "validation" in dataset.fold_dirs(fold_idx), "Validation set not found"

    validation_dataset = ComplexityDataset(
        dataset.fold_dirs(fold_idx)["validation"],
        dataset.fold_dirs(fold_idx)["validation_segmentations"],
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
