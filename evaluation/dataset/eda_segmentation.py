import os

import numpy as np
import quanproto.datasets.config_parser as quan_dataloader
from quanproto.eda import eda
from quanproto.utils.workspace import DATASET_DIR, EXPERIMENTS_PATH

config = {
    "dataset_dir": DATASET_DIR,
    # "dataset": "cub200",
    # "dataset": "dogs",
    "dataset": "cars196",
    # "dataset": "nico",
    # "dataset": "awa2",
}
# if the dataset name has a mini in it, we can use bars
use_bars = True if "mini" in config["dataset"] else False

dataset = quan_dataloader.get_dataset(config["dataset_dir"], config["dataset"])

log_dir = os.path.join(EXPERIMENTS_PATH, config["dataset"], "all", "segmentation")
os.makedirs(log_dir, exist_ok=True)


seg_methods = dataset._segmentation_info.keys()

for method in seg_methods:
    if method == "original":
        continue

    print(f"Method: {method}")
    num_masks = np.array(
        [len(entry["masks"]) for entry in dataset._segmentation_info[method].values()]
    )
    masks_dir = dataset._segmentation_dir

    sample_labels = np.array(list(dataset.sample_labels().values()))
    num_masks_x_labels = np.column_stack((num_masks, sample_labels))

    # make all statistics
    eda.segmentation_folder_statistics(
        num_masks_x_labels, masks_dir, log_dir, prefix=method, bars=use_bars
    )

    eda.segmentation_size_statistics(
        dataset.sample_info()["masks"][method]["size"],
        log_dir,
        prefix=method,
        bars=use_bars,
    )

    if "original" in seg_methods:
        eda.segmentation_object_overlap_statistics(
            dataset,
            method,
            log_dir,
            prefix=method,
            bars=use_bars,
        )
