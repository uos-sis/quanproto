import os

import quanproto.datasets.config_parser as quan_dataloader
from quanproto.utils.workspace import DATASET_DIR

config = {
    "dataset_dir": DATASET_DIR,
    # "dataset": "cub200",
    # "dataset": "dogs",
    "dataset": "cars196",
    # "dataset": "nico",  # you have to first download the dataste manually
    # "dataset": "awa2",
}

dataset = quan_dataloader.get_dataset(config["dataset_dir"], config["dataset"])

dataset.split_dataset(predefined=True)
