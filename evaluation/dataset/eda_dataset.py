import os

import numpy as np
import quanproto.datasets.config_parser as quan_dataloader
from quanproto.eda import eda
from quanproto.utils.workspace import DATASET_DIR, EXPERIMENTS_PATH

config = {
    "dataset_dir": DATASET_DIR,
    "dataset": "cub200",
    # "dataset": "dogs",
    # "dataset": "cars196",
    # "dataset": "nico",
    # "dataset": "awa2",
}

dataset = quan_dataloader.get_dataset(config["dataset_dir"], config["dataset"])

log_dir = os.path.join(EXPERIMENTS_PATH, config["dataset"], "all")
print(f"Saving logs to {log_dir}")
os.makedirs(log_dir, exist_ok=True)

sample_labels = np.array(list(dataset.sample_labels().values()))
sample_dir = dataset.sample_dir()

# make all statistics
eda.image_folder_statistics(sample_labels, sample_dir, log_dir, bars=False)

# K-Fold Split Statistics
split_statistics = eda.stratified_cross_validation_statistics(dataset.sample_labels())
eda.save_statistics(split_statistics, log_dir, "split_statistics")
