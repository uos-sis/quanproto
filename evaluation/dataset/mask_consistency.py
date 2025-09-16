from quanproto.utils.workspace import DATASET_DIR, EXPERIMENTS_PATH
import numpy as np
from quanproto.dataloader.custom import (
    make_mask_consistency_dataset,
)
from quanproto.techniques.mask_consistency import evaluate_mask_consistency

config = {
    "dataset_dir": DATASET_DIR,
    "dataset": "cub200_mini",
    "method": "original",
}

if __name__ == "__main__":
    dataset = make_mask_consistency_dataset(config, num_workers=0)

    class_mask_histograms = evaluate_mask_consistency(dataset)

    # compute the mean and std of the histograms
    mean_histograms = np.zeros(len(class_mask_histograms), dtype=np.float32)
    std_histograms = np.zeros(len(class_mask_histograms), dtype=np.float32)

    for class_id, histograms in enumerate(class_mask_histograms):
        mean_histograms[class_id] = histograms.mean(axis=0)
        std_histograms[class_id] = histograms.std(axis=0)

    # compute overall mean and std
    overall_mean = np.mean(mean_histograms)
    print(f"Overall mean: {overall_mean}")

    # mean std
    mean_std = np.mean(std_histograms)
    print(f"Overall std: {mean_std}")
