"""
This script is used to train the ProtoMask model using PyTorch Lightning.
"""

import argparse
import os
from copy import deepcopy
from typing import Any

import optuna
import quanproto.dataloader.segmentationmask as quan_dataloader
import quanproto.training.callbacks as quan_callbacks
import quanproto.utils.args as quan_args
import quanproto.utils.logger as quan_logger_functional
from optuna.integration import PyTorchLightningPruningCallback
from quanproto.logger import logger
from quanproto.models.config_parser import create_protomask
from quanproto.models.protomask.optimizer import make_optimizer
from quanproto.models.protomask.protomask import compute_loss
from quanproto.models.protomask.push import push_prototypes
from quanproto.training.scheduler import get_scheduler
from quanproto.utils import parameter
from quanproto.utils.workspace import WORKSPACE_PATH

import quanproto.training.training as quan_training

PROJECT_NAME = "ProtoMask"

# region Input arguments ---------------------------------------------------------------------------
quan_args.parser.add_argument(
    "--segmentation_method",
    type=str,
    default="sam",
    choices=[
        "sam2",
        "sam",
        "slit",
    ],
    help="Name of the augmentation pipeline to be used for training",
)

quan_args.parser.add_argument(
    "--fill_background_method",
    type=str,
    default="zero",
    choices=[
        "mean",
        "zero",
        "original",
    ],
)

quan_args.parser.add_argument(
    "--min_bbox_size",
    type=float,
    default=0.0,
    help="Minimum size of the bounding box compared to the image size",
)

args: argparse.Namespace = quan_args.parser.parse_args()


def train(trial: optuna.Trial | None = None, tune_config: dict[str, Any] | None = None) -> float:
    experiment_config = quan_args.make_experiment_config(args)

    parameter_config = deepcopy(
        parameter.protomask_params[
            experiment_config["dataset"] + "_" + experiment_config["segmentation_method"]
        ]
    )

    config_all = {**experiment_config, **parameter_config}

    if tune_config is not None:
        # override the config with the tune config
        config_all.update(tune_config)

    log_dir = quan_logger_functional.setup_logger(
        WORKSPACE_PATH,
        PROJECT_NAME,
        config_all,
    )
    model_dir = os.path.join(log_dir, "models", logger.name())
    os.makedirs(model_dir, exist_ok=True)

    config = logger.config()

    if config["min_bbox_size"] > 0.0:
        assert (
            config["fill_background_method"] == "original"
        ), "If you want to use the original image as background, you have to set min_bbox_size to 0.0"
    # region Dataset -------------------------------------------------------------------------------
    train_loader, validation_loader, push_loader, class_weights = (
        quan_dataloader.make_dataloader_trainingset(
            config,
            num_workers=config["num_workers"],
            crop=config["crop_input"],
            fill_background_method=config["fill_background_method"],
            min_bbox_size=config["min_bbox_size"],
        )
    )
    # endregion Dataset ----------------------------------------------------------------------------

    # region Model ---------------------------------------------------------------------------------
    model = create_protomask(
        feature=config["features"],
        pretrained=config["init_feature_weights"],
        num_classes=config["num_classes"],
        multi_label=config["multi_label"],
        prototypes_per_class=config["prototypes_per_class"],
        prototype_channel_num=config["prototype_channel_num"],
    )
    # endregion Model ------------------------------------------------------------------------------
    model.compile(
        coefs={
            "crs_ent": config["coefs.crs_ent"],
            "clst": config["coefs.clst"],
            "sep": config["coefs.sep"],
            "div": config["coefs.div"],
            "l1": config["coefs.l1"],
        },
        class_weights=class_weights[0],  # these are the train class weights
    )

    # region Optimizer -----------------------------------------------------------------------------
    warm_optimizer, joint_optimizer, fine_tune_optimizer = make_optimizer(model, config)
    # endregion Optimizer --------------------------------------------------------------------------

    # region Scheduler -----------------------------------------------------------------------------
    warm_lr_scheduler, joint_lr_scheduler, fine_tune_lr_scheduler = get_scheduler(
        warm_optimizer, joint_optimizer, fine_tune_optimizer, config
    )
    # endregion Scheduler --------------------------------------------------------------------------

    # region Training ------------------------------------------------------------------------------
    generalMetricsCallback = quan_callbacks.GeneralMetricsCallback()
    lightning_model = quan_training.LightningTrainerModel(
        model,
        compute_loss,
        warm_optimizer,
        joint_optimizer,
        fine_tune_optimizer,
        warm_lr_scheduler,
        joint_lr_scheduler,
        fine_tune_lr_scheduler,
        config["warmup_epochs"],
        config["joint_epochs"],
        config["fine_tune_epochs"],
        config["accumulate_batches"],
        metric_callbacks=[generalMetricsCallback],
        model_dir=model_dir,
    )

    monitor = "val_accuracy"
    save_callback = quan_callbacks.SaveCallback(
        monitor=monitor,
        mode="max",
    )
    push_callback = quan_callbacks.PushCallback(
        push_epochs=[config["warmup_epochs"] + config["joint_epochs"]],
        push_loader=push_loader,
        push_fn=push_prototypes,
    )

    # catch termination signals
    try:
        if tune_config == None:
            acc = quan_training.train_lightning(
                model=lightning_model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                callbacks=[push_callback, save_callback],
                eval_every_n_epochs=config["eval_every_n_epochs"],
                no_progress_bar=config["no_progress_bar"],
            )
        else:
            if trial is None:
                raise ValueError("Trial is None")

            prune_callback = PyTorchLightningPruningCallback(trial, monitor="val_accuracy")
            acc = quan_training.tune_lightning(
                model=lightning_model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                callbacks=[push_callback, save_callback, prune_callback],
            )
    # catch all errors
    except Exception as e:
        print("Terminate training due to error")
        logger.finish()
        raise e

    # endregion Training ---------------------------------------------------------------------------
    logger.finish()
    return acc


def objective(trial: optuna.Trial) -> float:
    tune_config = {
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 10, step=5),
        "warm_optimizer_lr": trial.suggest_float("warm_optimizer_lr", 1e-5, 3e-3, log=True),
        "joint_scheduler.step_size": trial.suggest_int("joint_scheduler.step_size", 15, 35, step=5),
        "joint_optimizer_lr.backbone": trial.suggest_float(
            "joint_optimizer_lr.backbone",
            1e-5,
            1e-4,
            log=True,
        ),
        "joint_optimizer_lr.prototype_vectors": trial.suggest_float(
            "joint_optimizer_lr.prototype_vectors",
            1e-3,
            5e-3,
            log=True,
        ),
        "fine_tune_optimizer_lr": trial.suggest_float(
            "fine_tune_optimizer_lr", 1e-5, 3e-3, log=True
        ),
    }
    acc = train(trial, tune_config)
    print(f"Accuracy: {acc}")
    return acc


if __name__ == "__main__":
    if not args.tune:
        train()
    else:
        # tuning with optuna
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=20, interval_steps=10),
        )
        print(f"Sampler is {study.sampler.__class__.__name__}")

        study.optimize(objective, n_trials=args.n_trials)
