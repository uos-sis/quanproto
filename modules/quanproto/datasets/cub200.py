import json
import os

import numpy as np

CUB_200_2011_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
CUB_200_2011_SEGMENTATION_URL = (
    "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz"
)

from quanproto.datasets import functional as F
from quanproto.datasets.interfaces import DatasetBase


class CUB200(DatasetBase):
    def __init__(self, dataset_dir, dataset_name: str = "CUB_200_2011"):
        # dataset_name = CUB_200_2011_URL.split("/")[-1].split(".")[0]
        super().__init__(dataset_dir, dataset_name)

        # Extra Dictionaries
        # key: sample_id (int) value: bounding box coordinates tuple (min_x, min_y, max_x, max_y)
        self._bounding_boxes = {}
        # key: sample_id (int) value: part locations dictionary key: part_id value: (x, y, visible)
        self._part_locs = {}
        # key: part_id (int) value: part_name (str)
        self._part_names = {}
        # key: sample_id (int) value: attribute dictionary key: attribute_id value: (is_present, certainty_id)
        self._sample_attributes = {}
        # key: attribute_id (int) value: attribute_name (str)
        self._attribute_names = {}

        segmentation_name = CUB_200_2011_SEGMENTATION_URL.split("/")[-1].split(".")[0]
        self._download_dataset(dataset_dir, segmentation_name)
        self._load_dataset_info()

        # load the splits if they already exist
        self._read_split_info()

    def _download_dataset(self, dataset_dir: os.path, segmentation_name: str):
        # check if the dataset_dir contains the CUB200-2011 dataset if not download and extract it
        if not os.path.isdir(self._root_dir):
            # download the dataset
            os.system(f"wget {CUB_200_2011_URL} -P {dataset_dir}")

            # rename the dataset folder to the dataset name
            os.rename(
                os.path.join(dataset_dir, CUB_200_2011_URL.split("/")[-1]),
                ".".join([self._root_dir, CUB_200_2011_URL.split(".")[-1]]),
            )

            # extract the dataset
            if CUB_200_2011_URL.split("/")[-1].split(".")[-1] == "tgz":
                os.system(f"tar -xzf {self._root_dir}.tgz -C {dataset_dir}")
                # remove the tar.gz file
                os.system(f"rm {self._root_dir}.tgz")

                # rename the dataset folder to the dataset name
                os.rename(
                    os.path.join(
                        dataset_dir, CUB_200_2011_URL.split("/")[-1].split(".")[0]
                    ),
                    self._root_dir,
                )
            else:
                raise ValueError(f"Dataset is not a tar.gz file.")

            # move the attributes.txt file in the attributes folder, i don't know why it is not in the attributes folder
            os.system(
                f"mv {os.path.join(dataset_dir, 'attributes.txt')} {os.path.join(self._root_dir, 'attributes', 'attributes.txt')}"
            )

            # rename images folder to sample        # self.image_root = img_root
            # self.image_path = info["paths"]s
            os.rename(os.path.join(self._root_dir, "images"), self._sample_dir)
            # remane classes.txt to class_names.txt
            os.rename(
                os.path.join(self._root_dir, "classes.txt"),
                os.path.join(self._root_dir, "class_names.txt"),
            )
            # subtract 1 from the class ids so that they start from 0
            class_names = []
            with open(os.path.join(self._root_dir, "class_names.txt"), "r") as f:
                class_names = [
                    f"{int(line.split()[0]) - 1} {line.split()[1]}\n" for line in f
                ]
            with open(os.path.join(self._root_dir, "class_names.txt"), "w") as f:
                f.writelines(class_names)

            # remane images.txt to sample_names.txt
            os.rename(
                os.path.join(self._root_dir, "images.txt"),
                os.path.join(self._root_dir, "sample_names.txt"),
            )
            # subtract 1 from the sample ids so that they start from 0
            sample_names = []
            with open(os.path.join(self._root_dir, "sample_names.txt"), "r") as f:
                sample_names = [
                    f"{int(line.split()[0]) - 1} {line.split()[1]}\n" for line in f
                ]
            with open(os.path.join(self._root_dir, "sample_names.txt"), "w") as f:
                f.writelines(sample_names)

            # remane image_class_labels.txt to sample_labels.txt
            os.rename(
                os.path.join(self._root_dir, "image_class_labels.txt"),
                os.path.join(self._root_dir, "sample_labels.txt"),
            )
            # subtract 1 from the sample ids so that they start from 0
            sample_labels = []
            with open(os.path.join(self._root_dir, "sample_labels.txt"), "r") as f:
                sample_labels = [
                    f"{int(line.split()[0]) -1} {int(line.split()[1]) -1}\n"
                    for line in f
                ]
            with open(os.path.join(self._root_dir, "sample_labels.txt"), "w") as f:
                f.writelines(sample_labels)

            # subtract 1 from the class ids so that they start from 0 in the train_test_split.txt file
            train_test_split = []
            with open(os.path.join(self._root_dir, "train_test_split.txt"), "r") as f:
                train_test_split = [
                    f"{int(line.split()[0]) - 1} {line.split()[1]}\n" for line in f
                ]
            with open(os.path.join(self._root_dir, "train_test_split.txt"), "w") as f:
                f.writelines(train_test_split)

            # subtract 1 from the sample ids so that they start from 0 in the bounding_boxes.txt file
            bounding_boxes = []
            with open(os.path.join(self._root_dir, "bounding_boxes.txt"), "r") as f:
                bounding_boxes = [
                    f"{int(line.split()[0]) - 1} {' '.join([str(int(float(line.split()[1]))), str(int(float(line.split()[2]))), str(int(float(line.split()[1]) + float(line.split()[3]))), str(int(float(line.split()[2]) + float(line.split()[4])))])}\n"
                    for line in f
                ]
            with open(os.path.join(self._root_dir, "bounding_boxes.txt"), "w") as f:
                f.writelines(bounding_boxes)

            # subtract 1 from the part_locs.txt and other files
            # file has the format <image_id> <part_id> <x> <y> <visible>
            with open(os.path.join(self._root_dir, "parts", "part_locs.txt"), "r") as f:
                part_locs = [
                    f"{int(line.split()[0]) - 1} {' '.join(line.split()[1:])}\n"
                    for line in f
                ]
            with open(os.path.join(self._root_dir, "parts", "part_locs.txt"), "w") as f:
                f.writelines(part_locs)

            # subtract 1 from the part_click_locs.txt file
            # file has the format <image_id> <part_id> <x> <y> <visible> <time>
            with open(
                os.path.join(self._root_dir, "parts", "part_click_locs.txt"), "r"
            ) as f:
                part_click_locs = [
                    f"{int(line.split()[0]) - 1} {' '.join(line.split()[1:])}\n"
                    for line in f
                ]
            with open(
                os.path.join(self._root_dir, "parts", "part_click_locs.txt"), "w"
            ) as f:
                f.writelines(part_click_locs)

            # subtract 1 from the image_attribute_labels.txt file
            # file has the format <image_id> <attribute_id> <is_present> <certainty_id> <time>
            with open(
                os.path.join(
                    self._root_dir, "attributes", "image_attribute_labels.txt"
                ),
                "r",
            ) as f:
                image_attribute_labels = [
                    f"{int(line.split()[0]) - 1} {' '.join(line.split()[1:])}\n"
                    for line in f
                ]
            with open(
                os.path.join(
                    self._root_dir, "attributes", "image_attribute_labels.txt"
                ),
                "w",
            ) as f:
                f.writelines(image_attribute_labels)

        # check if the dataset_dir contains the segmentation masks if not download and extract them
        if not os.path.isdir(os.path.join(self._root_dir, segmentation_name)):
            # download the segmentation masks
            print("Downloading segmentation masks...")
            os.system(f"wget {CUB_200_2011_SEGMENTATION_URL} -P {dataset_dir}")

            # extract the segmentation masks
            if CUB_200_2011_SEGMENTATION_URL.split("/")[-1].split(".")[-1] == "tgz":
                print("Extracting segmentation masks...")
                os.system(
                    f"tar -xzf {os.path.join(dataset_dir, segmentation_name)}.tgz -C {self._root_dir}"
                )
                # remove the tar.gz file
                os.system(f"rm {os.path.join(dataset_dir, segmentation_name)}.tgz")
            else:
                raise ValueError(f"Dataset {segmentation_name} is not a tar.gz file.")

            # rename the segmentation folder to segmentations if it is not already named
            if not os.path.isdir(os.path.join(self._root_dir, "segmentations")):
                os.rename(
                    os.path.join(self._root_dir, segmentation_name),
                    os.path.join(self._root_dir, "segmentations"),
                )

            # get all folder in the segmentation folder
            segmentation_folders = os.listdir(
                os.path.join(self._root_dir, segmentation_name)
            )

            # move all the segmentation folders to segmentations/original
            os.makedirs(os.path.join(self._root_dir, "segmentations", "original"))
            for folder in segmentation_folders:
                os.rename(
                    os.path.join(self._root_dir, "segmentations", folder),
                    os.path.join(self._root_dir, "segmentations", "original", folder),
                )

            # create segmentation info
            with open(os.path.join(self._root_dir, "sample_names.txt"), "r") as f:
                sample_names = {int(line.split()[0]): line.split()[1] for line in f}

            segmentation_info = {"original": {}}
            for sample_id, sample_name in sample_names.items():
                segmentation_name = sample_name.replace(".jpg", ".png")

                # remove .png from the end of the segmentation name
                segmentation_folder = ".".join(segmentation_name.split(".")[:-1])
                os.makedirs(
                    os.path.join(
                        self._root_dir, "segmentations", "original", segmentation_folder
                    )
                )

                new_segmentation_name = os.path.join(segmentation_folder, "mask_0.png")

                # rename the segmentation mask
                os.rename(
                    os.path.join(
                        self._root_dir, "segmentations", "original", segmentation_name
                    ),
                    os.path.join(
                        self._root_dir,
                        "segmentations",
                        "original",
                        new_segmentation_name,
                    ),
                )

                segmentation_info["original"][sample_id] = {"masks": {0: "mask_0.png"}}

            # save the segmentation names as a json file
            with open(os.path.join(self._root_dir, "segmentation_info.json"), "w") as f:
                json.dump(segmentation_info, f)

    def _load_dataset_info(self) -> None:
        super()._load_dataset_info()

        with open(os.path.join(self._root_dir, "bounding_boxes.txt"), "r") as f:
            self._bounding_boxes = {
                int(line.split()[0]): tuple(map(float, line.split()[1:])) for line in f
            }

        with open(os.path.join(self._root_dir, "parts", "part_locs.txt"), "r") as f:
            parts = f.readlines()
            for i in range(0, len(parts), 15):
                for j in range(0, 15):
                    # check the visibility of the part. last number in the line
                    if int(parts[i + j].split()[4]) == 1:
                        # check if the image id is already in the dictionary
                        if int(parts[i].split()[0]) not in self._part_locs:
                            self._part_locs[int(parts[i].split()[0])] = {
                                int(parts[i + j].split()[1]): (
                                    float(parts[i + j].split()[2]),
                                    float(parts[i + j].split()[3]),
                                )
                            }
                        else:
                            self._part_locs[int(parts[i].split()[0])][
                                int(parts[i + j].split()[1])
                            ] = (
                                float(parts[i + j].split()[2]),
                                float(parts[i + j].split()[3]),
                            )

        with open(os.path.join(self._root_dir, "parts", "parts.txt"), "r") as f:
            self._part_names = {int(line.split()[0]): line.split()[1] for line in f}

        with open(
            os.path.join(self._root_dir, "attributes", "image_attribute_labels.txt"),
            "r",
        ) as f:
            attributes = f.readlines()
            for i in range(0, len(attributes), 312):
                self._sample_attributes[int(attributes[i].split()[0])] = {
                    int(attributes[i + j].split()[1]): (
                        int(attributes[i + j].split()[2]),
                        int(attributes[i + j].split()[3]),
                    )
                    for j in range(1, 312)
                }

        with open(
            os.path.join(self._root_dir, "attributes", "attributes.txt"), "r"
        ) as f:
            self._attribute_names = {
                int(line.split()[0]): line.split()[1] for line in f
            }

    def make_mini_dataset(self, n_classes: int = 10) -> None:
        super().make_mini_dataset(n_classes)

        new_bounding_boxes = {}
        new_part_locs = {}
        new_sample_attributes = {}

        for key in self._sample_names.keys():
            new_bounding_boxes[key] = self._bounding_boxes[key]
            new_part_locs[key] = self._part_locs[key]
            new_sample_attributes[key] = self._sample_attributes[key]

        self._bounding_boxes = new_bounding_boxes
        self._part_locs = new_part_locs
        self._sample_attributes = new_sample_attributes

    def save_mini_dataset(self):
        super().save_mini_dataset()

        # delete bounding_boxes.txt
        if os.path.exists(os.path.join(self._root_dir, "bounding_boxes.txt")):
            os.remove(os.path.join(self._root_dir, "bounding_boxes.txt"))

        with open(os.path.join(self._root_dir, "bounding_boxes.txt"), "w") as f:
            for key, val in self._bounding_boxes.items():
                f.write(f"{key} {' '.join(map(str, val))}\n")

        # delete parts locs
        if os.path.exists(os.path.join(self._root_dir, "parts", "part_locs.txt")):
            os.remove(os.path.join(self._root_dir, "parts", "part_locs.txt"))

        with open(os.path.join(self._root_dir, "parts", "part_locs.txt"), "w") as f:
            for key, val in self._part_locs.items():
                for part_id in range(15):
                    if part_id in val:
                        f.write(
                            f"{key} {part_id} {' '.join(map(str, val[part_id]))} 1\n"
                        )
                    else:
                        f.write(f"{key} {part_id} 0.0 0.0 0\n")

        new_attributes = []
        with open(
            os.path.join(self._root_dir, "attributes", "image_attribute_labels.txt"),
            "r",
        ) as f:
            attributes = f.readlines()
            for i in range(len(self._sample_names)):
                for j in range(312):
                    new_attributes.append(attributes[i * 312 + j])

        os.remove(
            os.path.join(self._root_dir, "attributes", "image_attribute_labels.txt")
        )
        with open(
            os.path.join(self._root_dir, "attributes", "image_attribute_labels.txt"),
            "w",
        ) as f:
            f.writelines(new_attributes)

    def split_dataset(
        self,
        k: int = 2,
        seed: int = 42,
        shuffle: bool = True,
        stratified: bool = True,
        train_size: float = 0.7,
        predefined: bool = False,
    ) -> None:
        # self.decompress_samples_folder()
        super().split_dataset(k, seed, shuffle, stratified, train_size, predefined)
        # self.compress_samples_folder()

    def bounding_boxes(self):
        if len(self._bounding_boxes) == 0:
            raise ValueError("Bounding boxes have not been loaded.")
        return self._bounding_boxes

    def part_locations(self):
        return self._part_locs

    def part_names(self):
        return self._part_names

    def sample_attributes(self):
        return self._sample_attributes

    def attribute_names(self):
        return self._attribute_names

    def fold_info(self, k: int, dir_name) -> dict:
        info = super().fold_info(k, dir_name)

        return self._add_specific_info(info)

    def test_info(self) -> dict:
        info = super().test_info()

        return self._add_specific_info(info)

    def sample_info(self) -> dict:
        info = super().sample_info()

        return self._add_specific_info(info)

    def _add_specific_info(self, info: dict) -> dict:

        # add bounding boxes
        info["bboxes"] = [[self._bounding_boxes[img]] for img in info["ids"]]

        # add part locations
        info["partlocs"] = [self._part_locs[img] for img in info["ids"]]

        info["attributes"] = [self._sample_attributes[img] for img in info["ids"]]

        for id, path in zip(info["ids"], info["paths"]):
            masks_info = self._segmentation_info["original"][id]
            masks_paths = [
                os.path.join(
                    ".".join(path.split(".")[:-1]), masks_info["masks"][mask_id]
                )
                for mask_id in masks_info["masks"]
            ]

            if "masks" not in info:
                info["masks"] = {}
            if "original" not in info["masks"]:
                info["masks"]["original"] = {}
                info["masks"]["original"]["paths"] = []

            info["masks"]["original"]["paths"].append(masks_paths)

            for prop_key, prop_dic in masks_info.items():
                if prop_key == "masks":
                    continue
                if prop_key not in info["masks"]["original"]:
                    info["masks"]["original"][prop_key] = []
                info["masks"]["original"][prop_key].append(
                    [val for val in prop_dic.values()]
                )

        return info
