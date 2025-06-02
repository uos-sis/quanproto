import os

import numpy as np
import quanproto.datasets.config_parser as quan_dataloader
from quanproto.eda import eda
from quanproto.utils.workspace import DATASET_DIR, EXPERIMENTS_PATH


def save_image_statistics(output_dir: os.path, log_dir: os.path, prefix: str):

    counts, vals = eda.color_histogram(output_dir)
    norm_counts = counts / np.sum(counts, axis=1, keepdims=True)
    eda.save_color_histogram(norm_counts, vals, log_dir, f"{prefix}_color_histogram")

    statistics = eda.color_statistics(counts, vals)
    eda.save_statistics(statistics, log_dir, f"{prefix}_color_statistics")


config = {
    "dataset_dir": DATASET_DIR,
    "dataset": "cub200",
    # "dataset": "dogs",
    # "dataset": "cars196",
    # "dataset": "nico",
    # "dataset": "awa2",
}
dataset = quan_dataloader.get_dataset(config["dataset_dir"], config["dataset"])

log_dir = os.path.join(EXPERIMENTS_PATH, config["dataset"], "split")
os.makedirs(log_dir, exist_ok=True)


for fold_idx in range(dataset.num_folds()):

    log_fold_dir = os.path.join(log_dir, f"fold_{fold_idx}")
    os.makedirs(log_fold_dir, exist_ok=True)

    fold_names = []
    if dataset.fold_split(fold_idx)[0].size > 0:
        fold_names.append("train")
    if dataset.fold_split(fold_idx)[1].size > 0:
        fold_names.append("validation")

    for set_id in range(len(fold_names)):
        fold_info = dataset.fold_info(fold_idx, fold_names[set_id])
        class_labels = np.array(fold_info["labels"])

        fold_dir = dataset.fold_dirs(k=fold_idx)[fold_names[set_id]]
        fold_prefix = f"fold_{fold_idx}"

        os.makedirs(os.path.join(log_fold_dir, fold_names[set_id]), exist_ok=True)
        eda.image_folder_statistics(
            class_labels,
            fold_dir,
            os.path.join(log_fold_dir, fold_names[set_id]),
            fold_prefix,
        )

log_test_dir = os.path.join(log_dir, "test")
os.makedirs(log_test_dir, exist_ok=True)

test_info = dataset.test_info()
class_labels = np.array(test_info["labels"])
test_dir = dataset.test_dirs()["test"]
test_prefix = "test"

eda.image_folder_statistics(class_labels, test_dir, log_test_dir, test_prefix)
