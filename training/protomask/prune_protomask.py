import json
import os
import random

import numpy as np
import quanproto.dataloader.segmentationmask as quan_dataloader
import quanproto.training.callbacks as quan_callbacks
import quanproto.utils.logger as quan_logger_functional
import torch
from quanproto.explanations.config_parser import load_model
from quanproto.logger import logger
from quanproto.models.protomask.optimizer import make_optimizer
from quanproto.models.protomask.protomask import compute_loss
from quanproto.models.protomask.prune import prune_prototypes
from quanproto.training.scheduler import get_scheduler
from quanproto.utils.workspace import DATASET_DIR, EXPERIMENTS_PATH, WORKSPACE_PATH

import quanproto.training.training as quan_training
from quanproto.evaluation.folder_utils import get_run_info

experiment_config = {
    "experiment_dir": f"{EXPERIMENTS_PATH}/ProtoMask/dogs",
    "dataset_dir": DATASET_DIR,
}
PROJECT_NAME = "ProtoMaskPruned"
K_NEAREST_NEIGHBORS = 6
PRUNE_THRESHOLD = 3
NUM_WORKERS = 32
CROP = True
FILL_BACKGROUND_METHOD = "zero"
TRAINING_PHASE = "fine_tune"
FINE_TUNE_EPOCHS = 5
BATCH_SIZE = 8


def prune_model(run_dict, prune_fn, **prune_kwargs):
    # load the config file
    with open(run_dict["config"], "r") as f:
        run_config = json.load(f)

    seed = run_config["seed"]
    torch.manual_seed(seed)

    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # If you're using CUDA, you can also set the seed for GPUs
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_float32_matmul_precision("high")

    # check if the needed information to make a dataloader is in the config file
    if "dataset" not in run_config:
        raise KeyError("Could not find dataset in the config file")
    if "fold_idx" not in run_config:
        raise KeyError("Could not find fold_idx in the config file")

    run_config["dataset_dir"] = run_dict["dataset_dir"]
    dataloader = quan_dataloader.prune_dataloader(
        run_config,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        crop=CROP,
        fill_background_method=FILL_BACKGROUND_METHOD,
    )

    model = load_model(
        run_config,
        None,
        run_dict[TRAINING_PHASE],
    )

    model.cuda()
    num_prototypes = model.num_prototypes()
    prune_prototypes(dataloader, model=model, **prune_kwargs)
    new_num_prototypes = model.num_prototypes()
    pruned_num_prototypes = num_prototypes - new_num_prototypes
    print(
        f"Pruned prototypes from {num_prototypes} to {new_num_prototypes}, total pruned prototypes: {pruned_num_prototypes}"
    )
    return model


if __name__ == "__main__":
    run_info = get_run_info(experiment_config)
    for run_name, run_dict in run_info.items():
        run_dict["dataset_dir"] = experiment_config["dataset_dir"]

        print(f"Prune {run_name}")

        model = prune_model(
            run_dict,
            prune_prototypes,
            k=K_NEAREST_NEIGHBORS,
            prune_threshold=PRUNE_THRESHOLD,
        )

        with open(run_dict["config"], "r") as f:
            run_config = json.load(f)

        run_config["dataset_dir"] = run_dict["dataset_dir"]
        run_config["original_run_name"] = run_name
        run_config["warmup_epochs"] = 0
        run_config["joint_epochs"] = 0
        run_config["fine_tune_epochs"] = FINE_TUNE_EPOCHS
        run_config["logger"] = "tensorboard"
        run_config["num_prototypes"] = model.num_prototypes()

        print(f"dataset: {run_config['dataset']}")
        print(f"dataset dir: {run_config['dataset_dir']}")

        # INFO: This is the same as in the training script
        log_dir = quan_logger_functional.setup_logger(
            WORKSPACE_PATH,
            PROJECT_NAME,
            run_config,
        )
        model_dir = os.path.join(log_dir, "models", logger.name())
        os.makedirs(model_dir, exist_ok=True)

        run_config = logger.config()

        train_loader, validation_loader, _, class_weights = (
            quan_dataloader.make_dataloader_trainingset(
                run_config,
                num_workers=NUM_WORKERS,
                crop=CROP,
                fill_background_method=FILL_BACKGROUND_METHOD,
            )
        )

        model.compile(
            coefs={
                "crs_ent": run_config["coefs.crs_ent"],
                "clst": run_config["coefs.clst"],
                "sep": run_config["coefs.sep"],
                "div": run_config["coefs.div"],
                "l1": run_config["coefs.l1"],
            },
            class_weights=class_weights[0],  # these are the train class weights
        )

        # region Optimizer and Scheduler ---------------------------------------------------------------
        _, _, fine_tune_optimizer = make_optimizer(model, run_config)

        _, _, fine_tune_lr_scheduler = get_scheduler(None, None, fine_tune_optimizer, run_config)

        generalMetricsCallback = quan_callbacks.GeneralMetricsCallback()
        lightning_model = quan_training.LightningTrainerModel(
            model,
            compute_loss,
            None,
            None,
            fine_tune_optimizer,
            None,
            None,
            fine_tune_lr_scheduler,
            0,
            0,
            run_config["fine_tune_epochs"],
            metric_callbacks=[generalMetricsCallback],
            model_dir=model_dir,
        )

        monitor = "val_accuracy"
        save_callback = quan_callbacks.SaveCallback(
            monitor=monitor,
            mode="max",
        )

        acc = quan_training.train_lightning(
            model=lightning_model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            callbacks=[save_callback],
            eval_every_n_epochs=1,
            no_progress_bar=True,
        )

        logger.finish()
