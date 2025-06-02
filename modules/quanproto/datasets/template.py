import os
from quanproto.datasets.interfaces import DatasetBase


class CUB200(DatasetBase):
    def __init__(self, dataset_dir, dataset_name: str = "Template"):
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
            # download the dataset
            pass

    def _load_dataset_info(self) -> None:
        super()._load_dataset_info()
        # load extra information like bounding boxes, part locations, attributes
        # ...

    def fold_info(self, k: int, dir_name) -> dict:
        info = super().fold_info(k, dir_name)

        return self._add_specific_info(info)

    def test_info(self) -> dict:
        info = super().test_info()

        return self._add_specific_info(info)

    def _add_specific_info(self, info: dict) -> dict:

        # for example add bounding boxes, part locations, attributes
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

        # add part locations
        info["partlocs"] = [self._part_locs[img] for img in info["ids"]]

        info["attributes"] = [self._sample_attributes[img] for img in info["ids"]]

        return info
