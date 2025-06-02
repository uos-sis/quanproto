import argparse
from copy import deepcopy

import torch

from quanproto.utils.workspace import DATASET_DIR

# region Input arguments ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Description of your program.")

# Add arguments
parser.add_argument(
    "--features",
    type=str,
    default="efficientnet-b0",
    help="Name of the backbone feature extractor",
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=DATASET_DIR,
    help="Path to the dataset",
)
parser.add_argument("--dataset", type=str, default="cub200", help="Name of the dataset")
parser.add_argument(
    "--fold_idx", type=int, default=0, help="Index of the fold to be used for training"
)
parser.add_argument(
    "--augmentation_pipeline",
    type=str,
    default="geometric_photometric",
    help="Name of the augmentation pipeline to be used for training",
)
parser.add_argument(
    "--augmentation_range",
    type=str,
    default="medium_medium",
    help="Name of the augmentation ranges to be used for training",
)
parser.add_argument(
    "--logger",
    type=str,
    default="tensorboard",
    choices=["normal", "tensorboard"],
    help="Name of the logger to be used for logging",
)
parser.add_argument(
    "--crop_input",
    action="store_true",
    help="Crop the image to the bounding box if bounding box is available",
)
parser.add_argument(
    "--eval_every_n_epochs",
    type=int,
    default=1,
    help="Evaluate the model every n epochs",
)
parser.add_argument(
    "--seed",
    type=int,
    # default=np.random.randint(0, 1000),
    default=4861,
    help="Seed to be used for reproducibility",
)
parser.add_argument("--tune", action="store_true", help="tune the model using ray tune")
parser.add_argument(
    "--num_workers",
    type=int,
    default=16,
    help="Number of workers to be used for dataloading",
)
parser.add_argument(
    "--no_progress_bar",
    action="store_true",
    help="Do not show progress bar during training",
)
parser.add_argument(
    "--n_trials",
    type=int,
    default=100,
    help="Number of trials to be used for tuning (single process only)",
)


def make_experiment_config(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    config = deepcopy(vars(args))

    if isinstance(config["augmentation_pipeline"], list) or isinstance(
        config["augmentation_pipeline"], tuple
    ):
        return config

    aug_pipeline_keys: list[str] = config["augmentation_pipeline"].split("_")
    aug_pipeline_ranges: list[str] = config["augmentation_range"].split("_")

    del config["augmentation_pipeline"]
    del config["augmentation_range"]

    # make a list of keys and ranges
    augmentations = []
    for key, range_id in zip(aug_pipeline_keys, aug_pipeline_ranges):
        augmentations.append((key, range_id))
    config["augmentation_pipeline"] = augmentations

    return config
