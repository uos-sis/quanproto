import os
import numpy as np
from PIL import Image
import deeplake

from quanproto.datasets.interfaces import DatasetBase


class Cars196(DatasetBase):
    def __init__(self, dataset_dir, dataset_name="cars196") -> None:
        super().__init__(dataset_dir, dataset_name)

        # Extra Dictionaries
        # key: sample_id (int) value: bounding box coordinates tuple (x, y, width, height) (float)
        self._bounding_boxes = {}

        self._download_dataset()

        self._load_dataset_info()
        self._read_split_info()

    def _download_dataset(self) -> None:
        # create the dataset directory and image directory if they don't exist
        if not os.path.isdir(self._root_dir):
            os.makedirs(self._root_dir)
        else:
            # already downloaded
            return

        # make image directory
        os.makedirs(self._sample_dir)
        # make class folder
        for i in range(0, 196):
            class_id_str = str(i).zfill(3)
            self._class_names[i] = class_id_str
            if not os.path.isdir(os.path.join(self._sample_dir, class_id_str)):
                os.makedirs(os.path.join(self._sample_dir, class_id_str))

        train_ds = deeplake.load("hub://activeloop/stanford-cars-train")
        test_ds = deeplake.load("hub://activeloop/stanford-cars-test")
        all_ds = [train_ds, test_ds]

        image_id = 0
        train_test_split = [list(), list()]
        for i, ds in enumerate(all_ds):
            for _, item in enumerate(ds):
                image_tensor = item["images"]
                image_arr = image_tensor.numpy()

                class_id = item["car_models"].numpy()[0]
                image_name = f"{self._class_names[class_id]}_{str(image_id).zfill(5)}.jpg"

                self._sample_names[image_id] = os.path.join(self._class_names[class_id], image_name)
                self._sample_labels[image_id] = class_id

                bb = item["boxes"].numpy()[0]
                self._bounding_boxes[image_id] = (bb[0], bb[1], bb[2], bb[3])

                # check if the image is grayscale, some images are grayscale i don't know why
                if image_arr.shape[2] == 1:
                    rgb_image = np.repeat(image_arr, 3, axis=2)
                    image = Image.fromarray(rgb_image, mode="RGB")
                    image.save(os.path.join(self._sample_dir, self._sample_names[image_id]))
                else:
                    image = Image.fromarray(image_arr, mode="RGB")
                    image.save(os.path.join(self._sample_dir, self._sample_names[image_id]))

                train_test_split[i].append(image_id)
                image_id += 1
                if image_id % 100 == 0:
                    print(f"Downloaded {image_id} images")

        with open(os.path.join(self._root_dir, "train_test_split.txt"), "w") as f:
            # <image_id> <is_train>
            for img_id in train_test_split[0]:
                f.write(f"{img_id} 1\n")
            for img_id in train_test_split[1]:
                f.write(f"{img_id} 0\n")

        with open(os.path.join(self._root_dir, "sample_names.txt"), "w") as f:
            for image_id in self._sample_names:
                f.write(f"{image_id} {self._sample_names[image_id]}\n")

        with open(os.path.join(self._root_dir, "sample_labels.txt"), "w") as f:
            for image_id in self._sample_labels:
                f.write(f"{image_id} {self._sample_labels[image_id]}\n")

        with open(os.path.join(self._root_dir, "class_names.txt"), "w") as f:
            for class_id in self._class_names:
                f.write(f"{class_id} {self._class_names[class_id]}\n")

        with open(os.path.join(self._root_dir, "bounding_boxes.txt"), "w") as f:
            for image_id in self._bounding_boxes:
                bb_str = " ".join([str(bb) for bb in self._bounding_boxes[image_id]])
                f.write(f"{image_id} {bb_str}\n")

    def _load_dataset_info(self) -> None:
        super()._load_dataset_info()

        with open(os.path.join(self._root_dir, "bounding_boxes.txt"), "r") as f:
            self._bounding_boxes = {
                int(line.split()[0]): tuple(map(float, line.split()[1:])) for line in f
            }

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

    def fold_info(self, k: int, dir_name) -> dict:
        info = super().fold_info(k, dir_name)

        return self._add_specific_info(info)

    def test_info(self) -> dict:
        info = super().test_info()

        return self._add_specific_info(info)

    def _add_specific_info(self, info: dict) -> dict:

        # add bounding boxes
        info["bboxes"] = [
            [
                [
                    self._bounding_boxes[img][0],
                    self._bounding_boxes[img][1],
                    self._bounding_boxes[img][0] + self._bounding_boxes[img][2],
                    self._bounding_boxes[img][1] + self._bounding_boxes[img][3],
                ]
            ]
            for img in info["ids"]
        ]

        return info

    def bounding_boxes(self):
        if len(self._bounding_boxes) == 0:
            raise ValueError("Bounding boxes have not been loaded.")
        return self._bounding_boxes
