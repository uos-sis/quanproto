import os
import shutil
import numpy as np
import scipy.io
from quanproto.datasets.interfaces import DatasetBase
import xml.etree.ElementTree as ET

STANFORD_DOGS_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
ANNOTATIONS_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
SPLIT_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"


class DOGS(DatasetBase):
    def __init__(self, dataset_dir, dataset_name: str = "dogs"):
        super().__init__(dataset_dir, dataset_name)

        # Extra Dictionaries
        # key: sample_id (int) value: bounding box coordinates tuple (x, y, width, height) (float)
        self._bounding_boxes = {}
        # key: sample_id (int) value: part locations dictionary key: part_id value: (x, y, visible)
        self._part_locs = {}
        # key: part_id (int) value: part_name (str)
        self._part_names = {}
        # key: sample_id (int) value: attribute dictionary key: attribute_id value: (is_present, certainty_id)
        self._sample_attributes = {}
        # key: attribute_id (int) value: attribute_name (str)
        self._attribute_names = {}

        self._download_dataset(dataset_dir)
        self._load_dataset_info()

        # load the splits if they already exist
        self._read_split_info()

    def _download_dataset(self, dataset_dir):
        # check if the dataset_dir contains the CUB200-2011 dataset if not download and extract it
        if not os.path.isdir(self._root_dir):
            os.makedirs(self._root_dir)
            # download the dataset
            os.system(f"wget {STANFORD_DOGS_URL} -P {dataset_dir}")
            os.system(f"wget {ANNOTATIONS_URL} -P {dataset_dir}")
            os.system(f"wget {SPLIT_URL} -P {dataset_dir}")

            # extract the dataset
            os.system(f"tar -xvf {dataset_dir}/images.tar -C {dataset_dir}")
            os.system(f"tar -xvf {dataset_dir}/annotation.tar -C {dataset_dir}")

            os.makedirs(f"{self._root_dir}/splits/")
            os.system(f"tar -xvf {dataset_dir}/lists.tar -C {self._root_dir}/splits/")

            os.system(f"rm {dataset_dir}/images.tar")
            os.system(f"rm {dataset_dir}/annotation.tar")
            os.system(f"rm {dataset_dir}/lists.tar")

            # move the dataset to the root directory
            os.system(f"mv {dataset_dir}/Images {self._root_dir}/samples/")
            os.system(f"mv {dataset_dir}/Annotation {self._root_dir}/annotations/")

            # create a class_names.txt file <class_id> <class_name>
            class_names = [class_name for class_name in os.listdir(f"{self._root_dir}/samples/")]
            class_names.sort()
            with open(f"{self._root_dir}/class_names.txt", "w") as f:
                for i, class_name in enumerate(class_names):
                    f.write(f"{i} {class_name}\n")

            train_mat = scipy.io.loadmat(f"{self._root_dir}/splits/train_list.mat")
            test_mat = scipy.io.loadmat(f"{self._root_dir}/splits/test_list.mat")

            sample_names = []
            sample_labels = []
            bounding_boxes = []
            train_test_split = []

            def read_sample(file_mat, idx, is_train):
                sample_name = file_mat["file_list"][idx][0][0]
                annotation_name = file_mat["annotation_list"][idx][0][0]
                label = file_mat["labels"][idx][0] - 1

                class_name = sample_name.split("/")[0]
                assert class_name == class_names[label]

                sample_names.append(sample_name)
                sample_labels.append(label)
                train_test_split.append(is_train)

                tree = ET.parse(f"{self._root_dir}/annotations/{annotation_name}")

                root = tree.getroot()
                xmin = int(root.find(".//bndbox/xmin").text)
                ymin = int(root.find(".//bndbox/ymin").text)
                xmax = int(root.find(".//bndbox/xmax").text)
                ymax = int(root.find(".//bndbox/ymax").text)

                bounding_boxes.append((xmin, ymin, xmax, ymax))

            for idx in range(train_mat["file_list"].size):
                read_sample(train_mat, idx, is_train=1)

            for idx in range(test_mat["file_list"].size):
                read_sample(test_mat, idx, is_train=0)

            # make np arrays
            sample_names = np.array(sample_names)
            sample_labels = np.array(sample_labels)
            bounding_boxes = np.array(bounding_boxes)
            train_test_split = np.array(train_test_split)
            sorted_indices = np.argsort(sample_names)

            sample_names = sample_names[sorted_indices]
            sample_labels = sample_labels[sorted_indices]
            bounding_boxes = bounding_boxes[sorted_indices]
            train_test_split = train_test_split[sorted_indices]

            # create a sample_names.txt file <sample_id> <sample_name>
            with open(f"{self._root_dir}/sample_names.txt", "w") as f:
                for i, sample_name in enumerate(sample_names):
                    f.write(f"{i} {sample_name}\n")

            # create a sample_labels.txt file <sample_id> <class_id>
            with open(f"{self._root_dir}/sample_labels.txt", "w") as f:
                for i, label in enumerate(sample_labels):
                    f.write(f"{i} {label}\n")

            # create a bounding_boxes.txt file <sample_id> <x> <y> <width> <height>
            with open(f"{self._root_dir}/bounding_boxes.txt", "w") as f:
                for i, bbox in enumerate(bounding_boxes):
                    f.write(f"{i} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

            # create a train_test_split.txt file <sample_id> <is_train>
            with open(f"{self._root_dir}/train_test_split.txt", "w") as f:
                for i, is_train in enumerate(train_test_split):
                    f.write(f"{i} {is_train}\n")

            # delete the splits dir
            shutil.rmtree(f"{self._root_dir}/splits/")

    def _load_dataset_info(self) -> None:
        super()._load_dataset_info()
        with open(os.path.join(self._root_dir, "bounding_boxes.txt"), "r") as f:
            self._bounding_boxes = {
                int(line.split()[0]): tuple(map(float, line.split()[1:])) for line in f
            }

    def fold_info(self, k: int, dir_name) -> dict:
        info = super().fold_info(k, dir_name)

        return self._add_specific_info(info)

    def test_info(self) -> dict:
        info = super().test_info()

        return self._add_specific_info(info)

    def make_mini_dataset(self, n_classes: int = 10) -> None:
        super().make_mini_dataset(n_classes)

        new_bounding_boxes = {}

        for key in self._sample_names.keys():
            new_bounding_boxes[key] = self._bounding_boxes[key]

        self._bounding_boxes = new_bounding_boxes

    def sample_info(self) -> dict:
        info = super().sample_info()

        return self._add_specific_info(info)

    def save_mini_dataset(self):
        super().save_mini_dataset()

        # delete bounding_boxes.txt
        if os.path.exists(os.path.join(self._root_dir, "bounding_boxes.txt")):
            os.remove(os.path.join(self._root_dir, "bounding_boxes.txt"))

        with open(os.path.join(self._root_dir, "bounding_boxes.txt"), "w") as f:
            for key, val in self._bounding_boxes.items():
                f.write(f"{key} {' '.join(map(str, val))}\n")

        # delete all folder of annotations not in class_names
        for folder in os.listdir(os.path.join(self._root_dir, "annotations")):
            if folder not in self._class_names.values():
                shutil.rmtree(os.path.join(self._root_dir, "annotations", folder))

    def _add_specific_info(self, info: dict) -> dict:

        # for example add bounding boxes, part locations, attributes
        # add bounding boxes
        info["bboxes"] = [
            [
                [
                    self._bounding_boxes[img][0],
                    self._bounding_boxes[img][1],
                    self._bounding_boxes[img][2],
                    self._bounding_boxes[img][3],
                ]
            ]
            for img in info["ids"]
        ]

        return info


if __name__ == "__main__":
    USER = os.getenv("USER")
    dataset = DOGS(f"/home/{USER}/data/quanproto")
