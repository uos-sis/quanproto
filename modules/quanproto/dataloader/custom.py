import skimage as ski
from albumentations.pytorch import ToTensorV2
from quanproto.features.config_parser import feature_dict
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
import os
import albumentations as A

from quanproto.augmentation import enums
from quanproto.datasets import functional as F

import quanproto.dataloader.params as quan_params
import quanproto.datasets.config_parser as quan_dataloader


class ImageIdxDataset(Dataset):
    def __init__(self, sample_dir: str, info) -> None:
        self.sample_dir = sample_dir

        self.paths = info["paths"]

        self.combine_bbs = False

        if "bboxes" in info:
            self.bboxes = info["bboxes"]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = ski.io.imread(os.path.join(self.sample_dir, self.paths[idx]))
        if len(img.shape) == 2:
            # convert to 3 channels
            img = ski.color.gray2rgb(img)
        if img.shape[2] == 4:
            img = (ski.color.rgba2rgb(img) * 255).astype(np.uint8)
        if img.shape[2] == 2:
            raise ValueError("Image has 2 channels")

        bbox = []
        if hasattr(self, "bboxes"):
            if self.combine_bbs:
                bbox = F.combine_bounding_boxes(self.bboxes[idx])
            else:
                # random is the correct one for the patches segmentation
                bbox = F.get_random_bounding_box(self.bboxes[idx])

        return img, bbox, idx


class MaskConsistencyDataset(Dataset):
    def __init__(
        self,
        root,
        info,
        num_classes,
        mask_method,
    ) -> None:
        self.root = os.path.join(root, mask_method)
        self.paths = info["masks"][mask_method]["paths"]
        self.labels = info["labels"]
        self.num_classes = num_classes

        self.max_partlocs = 0
        if "partlocs" in info:
            self.partlocs = info["partlocs"]
            # get the maximum number of keys in a partlocs dict
            for partloc in self.partlocs:
                self.max_partlocs = max(self.max_partlocs, len(partloc.keys()))

        if hasattr(self, "partlocs"):
            self.trans_pip = A.Compose(
                [
                    A.Resize(224, 224),  # so we can make a tensor of the masks
                    ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(format="xy"),
            )
        else:
            # error we need partlocs
            raise ValueError("No partlocs provided")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        # make a tensor of labels
        label = torch.tensor(self.labels[idx][0]).long()

        tf_masks = torch.empty(0)

        # use all available masks
        num_masks = len(self.paths[idx])

        for i in range(num_masks):
            mask_path = self.paths[idx][i]
            original_mask = ski.io.imread(os.path.join(self.root, mask_path))

            if len(original_mask.shape) == 3:
                if original_mask.shape[2] == 4:
                    original_mask = (ski.color.rgba2rgb * 255).astype(np.uint8)
                    original_mask = ski.color.rgb2gray(original_mask)
                elif original_mask.shape[2] == 2:
                    original_mask = original_mask[:, :, 0]
                else:
                    original_mask = ski.color.rgb2gray(original_mask)

            assert len(original_mask.shape) == 2, "Mask is not 2D"
            original_mask = original_mask > 0
            original_mask = original_mask.astype(np.uint8)

            height, width = original_mask.shape

            # partlocs = [p for p in self.partlocs[idx].values()]
            partlocs = [
                (p[0] if p[0] < width else width - 1, p[1] if p[1] < height else height - 1)
                for p in self.partlocs[idx].values()
            ]
            partlocs_ids = [p for p in self.partlocs[idx].keys()]

            for i, point in enumerate(partlocs):
                # check if the point is within the image bounds
                if point[0] < 0 or point[0] >= width or point[1] < 0 or point[1] >= height:
                    print(f"Mask path: {mask_path}")
                    raise ValueError(
                        f"Partloc {point} is out of bounds for image of size {height}x{width}"
                    )

            # first transform the keypoints based on the image
            t = self.trans_pip(image=original_mask, keypoints=partlocs)
            tf_partlocs = t["keypoints"]
            tf_mask = t["image"]

            # check what dtype is returned
            if isinstance(tf_mask, torch.Tensor):
                tf_mask = tf_mask.float()
            else:
                tf_mask = torch.tensor(tf_mask).float()
            tf_masks = torch.cat((tf_masks, tf_mask.unsqueeze(0)), 0)

        # make a tensor with (max_partlocs, 2) and fill it with -1
        t_partlocs = torch.full((self.max_partlocs, 2), -1, dtype=torch.float32)
        t_partlocs_ids = torch.full((self.max_partlocs, 1), -1, dtype=torch.int32)

        # overwrite the -1 entries with the actual partlocs
        for i, point in enumerate(tf_partlocs):
            t_partlocs[i] = torch.tensor([point[0], point[1]], dtype=torch.float32)
            t_partlocs_ids[i] = torch.tensor(partlocs_ids[i], dtype=torch.int32)

        return (
            tf_masks,
            t_partlocs,
            t_partlocs_ids,
            label,
        )

    def get_item_params(self, idx: int):
        mask_root = self.root
        mask_paths = self.paths[idx]
        partlocs = self.partlocs[idx]
        max_partlocs = self.max_partlocs
        label = torch.tensor(self.labels[idx][0]).long()

        return (mask_root, mask_paths, partlocs, max_partlocs, label)


def make_imageidx_dataloader(
    config: dict,
    batch_size=1,
    num_workers=quan_params.NUM_DATALOADER_WORKERS,
    pin_memory=quan_params.PIN_MEMORY,
):

    dataset = quan_dataloader.get_dataset(config["dataset_dir"], config["dataset"])

    imageidx_dataset = ImageIdxDataset(
        dataset.sample_dir(),
        dataset.sample_info(),
    )

    dataloader = DataLoader(
        imageidx_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataloader


def make_mask_consistency_dataset(
    config: dict,
    batch_size=1,
    num_workers=quan_params.NUM_DATALOADER_WORKERS,
    pin_memory=quan_params.PIN_MEMORY,
):
    dataset = quan_dataloader.get_dataset(
        config["dataset_dir"], config["dataset"], dataset_name=config["dataset"]
    )
    num_classes = dataset.num_classes()

    mask_dataset = MaskConsistencyDataset(
        dataset.segmentation_dir(),
        dataset.sample_info(),
        num_classes,
        config["method"],
    )

    return mask_dataset


def make_mask_consistency_dataloader(
    config: dict,
    batch_size=1,
    num_workers=quan_params.NUM_DATALOADER_WORKERS,
    pin_memory=quan_params.PIN_MEMORY,
):
    dataset = quan_dataloader.get_dataset(config["dataset_dir"], config["dataset"])
    num_classes = dataset.num_classes()

    mask_dataset = MaskConsistencyDataset(
        dataset.segmentation_dir(),
        dataset.sample_info(),
        num_classes,
        config["method"],
    )

    dataloader = DataLoader(
        mask_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataloader
