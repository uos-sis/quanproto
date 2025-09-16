import argparse
import os
import random
from copy import deepcopy
from typing import Any

import numpy as np
import optuna
import quanproto.dataloader.segmentationmask as quan_dataloader
import quanproto.datasets.config_parser as quan_dataloader
import quanproto.segmentation.config_parser as quan_config
import torch
from quanproto.dataloader.custom import ImageIdxDataset, make_mask_consistency_dataset
from quanproto.logger import logger
from quanproto.segmentation.segmentation import generate_segment_masks
from quanproto.techniques.mask_consistency import evaluate_mask_consistency
from quanproto.utils.workspace import DATASET_DIR, WORKSPACE_PATH
from torch.utils.data import DataLoader

PROJECT_NAME = "TuneSAM2Segmentation"

parser = argparse.ArgumentParser(description="Description of your program.")

# region Input arguments ---------------------------------------------------------------------------
parser.add_argument(
    "--method",
    type=str,
    default="sam2",
    choices=[
        "sam",
        "sam2",
        "slit",
    ],
    help="Name of the augmentation pipeline to be used for training",
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=DATASET_DIR,
    help="Path to the dataset",
)
parser.add_argument("--dataset", type=str, default="cub200", help="Name of the dataset")
parser.add_argument(
    "--seed",
    type=int,
    # default=np.random.randint(0, 1000),
    default=4861,
    help="Seed to be used for reproducibility",
)
parser.add_argument(
    "--n_trials",
    type=int,
    default=1,
    help="Number of trials to be used for tuning (single process only)",
)

args: argparse.Namespace = parser.parse_args()


def train(trial: optuna.Trial | None = None, tune_config: dict[str, Any] | None = None) -> float:
    torch.manual_seed(args.seed)

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # If you're using CUDA, you can also set the seed for GPUs
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_float32_matmul_precision("high")

    config = deepcopy(vars(args))
    config: dict[str, Any] = {**config, **quan_config.model_config_dic[args.method]}

    config.update(tune_config)

    # region Logging -------------------------------------------------------------------------------
    logger.init(
        project=PROJECT_NAME,
        config=config,
        log_type="tensorboard",
        verbose=0,
        force_new=True,
    )
    run_name: str = logger.name()
    logger.info("Run name: {}".format(run_name))
    # endregion Logging ----------------------------------------------------------------------------

    config = logger.config()

    log_subdir: str = os.path.join(
        config["dataset"],
        "segmentation_tuning",
        config["method"],
    )

    log_dir: str = os.path.join(WORKSPACE_PATH, "experiments", logger.project(), log_subdir)
    os.makedirs(log_dir, exist_ok=True)
    logger.set_log_dir(log_dir)

    dataset = quan_dataloader.get_dataset(
        config["dataset_dir"], config["dataset"], dataset_name=args.dataset
    )
    imageidx_dataset = ImageIdxDataset(
        dataset.sample_dir(),
        dataset.sample_info(),
    )

    dataloader = DataLoader(
        imageidx_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    # generate_segment_masks(dataset, dataloader, config)

    dataset = make_mask_consistency_dataset(config, num_workers=0)

    class_mask_histograms = evaluate_mask_consistency(dataset)

    # compute the mean and std of the histograms
    mean_histograms = np.zeros(len(class_mask_histograms), dtype=np.float32)
    std_histograms = np.zeros(len(class_mask_histograms), dtype=np.float32)

    for class_id, histograms in enumerate(class_mask_histograms):
        mean_histograms[class_id] = histograms.mean(axis=0)
        std_histograms[class_id] = histograms.std(axis=0)

    # compute overall mean and std
    overall_mean = np.mean(mean_histograms) * 100

    logger.log({"consistency": overall_mean})
    logger.finish()

    return overall_mean


def objective(trial: optuna.Trial) -> float:
    # sam2
    tune_config = {
        "points_per_side": trial.suggest_categorical("points_per_side", [16, 32, 64, 128]),
        "crop_n_layers": trial.suggest_categorical("crop_n_layers", [0, 1, 2]),
        "crop_n_points_downscale_factor": trial.suggest_categorical(
            "crop_n_points_downscale_factor", [1, 2, 4]
        ),
    }

    # slit
    # tune_config = {
    #     "threshold": trial.suggest_float("threshold", 1e-3, 1e-1, log=True),
    # }

    consistency = train(trial, tune_config)
    print(f"Consistency: {consistency} %")
    return consistency


if __name__ == "__main__":
    # tuning with optuna
    study = optuna.create_study(
        direction="maximize",
    )
    print(f"Sampler is {study.sampler.__class__.__name__}")

    study.optimize(objective, n_trials=args.n_trials)
