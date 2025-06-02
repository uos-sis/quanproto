import json
import os

import lightning as L
import torch
from lightning.pytorch import Trainer

import quanproto.metrics.general
import quanproto.metrics.helpers as helpers

general_metrics = {
    "accuracy": quanproto.metrics.general.accuracy,
    "precision": quanproto.metrics.general.precision,
    "recall": quanproto.metrics.general.recall,
    "f1_score": quanproto.metrics.general.f1_score,
}


class GeneralMetricsCallback(L.Callback):
    def __init__(self, metric_fn: dict = general_metrics):
        self.metric_fn = metric_fn

    def on_train_batch_end(
        self,
        pl_module: L.LightningModule,
        outputs,
        targets,
    ) -> None:

        logits = outputs[0]
        binary = False
        if pl_module.model.multi_label:
            mean_tp_activation = quanproto.metrics.general.mean_tp_activation(
                logits, targets
            )
            mean_tn_activation = quanproto.metrics.general.mean_tn_activation(
                logits, targets
            )

            threshold = (mean_tp_activation + mean_tn_activation) / 2
            pl_module.log(
                "train_threshold",
                threshold,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

            logits = helpers.label_prediction(logits, True, threshold)
            binary = True

        for metric_name, metric_fn in self.metric_fn.items():
            # logits should be the first output
            val = metric_fn(logits, targets, binary=binary, balanced=True)

            metric_name = f"train_{metric_name}"
            pl_module.log(
                metric_name, val, on_step=False, on_epoch=True, sync_dist=True
            )

    def on_validation_batch_end(
        self, pl_module: L.LightningModule, outputs, targets
    ) -> None:
        logits = outputs[0]
        binary = False
        if pl_module.model.multi_label:
            mean_tp_activation = quanproto.metrics.general.mean_tp_activation(
                logits, targets
            )
            mean_tn_activation = quanproto.metrics.general.mean_tn_activation(
                logits, targets
            )

            threshold = (mean_tp_activation + mean_tn_activation) / 2
            pl_module.log(
                "val_threshold", threshold, on_step=False, on_epoch=True, sync_dist=True
            )

            logits = helpers.label_prediction(logits, True, threshold)
            binary = True

        for metric_name, metric_fn in self.metric_fn.items():
            # logits should be the first output
            val = metric_fn(logits, targets, binary=binary)

            metric_name = f"val_{metric_name}"
            pl_module.log(
                metric_name, val, on_step=False, on_epoch=True, sync_dist=True
            )


class CompactnessMetricsCallback(L.Callback):
    def on_train_batch_end(
        self,
        pl_module: L.LightningModule,
        outputs,
        targets,
    ) -> None:
        # compute compactness
        pl_module.model.eval()
        with torch.no_grad():
            epsilon = 1e-3
            classification_sparsity = pl_module.model.classification_sparsity(epsilon)
            negative_positive_ratio = pl_module.model.negative_positive_reasoning_ratio(
                epsilon
            )
        pl_module.model.train()

        pl_module.log(
            "train_classification_sparsity",
            classification_sparsity,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        pl_module.log(
            "train_negative_positive_ratio",
            negative_positive_ratio,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_validation_batch_end(
        self, pl_module: L.LightningModule, outputs, targets
    ) -> None:
        # compute compactness
        pl_module.model.eval()
        with torch.no_grad():
            epsilon = 1e-3
            classification_sparsity = pl_module.model.classification_sparsity(epsilon)
            negative_positive_ratio = pl_module.model.negative_positive_reasoning_ratio(
                epsilon
            )

        pl_module.log(
            "val_classification_sparsity",
            classification_sparsity,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        pl_module.log(
            "val_negative_positive_ratio",
            negative_positive_ratio,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )


class PushCallback(L.Callback):
    def __init__(self, push_epochs, push_loader, push_fn, log_projection_images=False):
        self.push_loader = push_loader
        self.push_epochs = push_epochs
        self.push_fn = push_fn

    def on_train_epoch_start(self, trainer, pl_module):
        if pl_module.current_epoch in self.push_epochs:
            pl_module.model.eval()
            log = self.push_fn(
                pl_module.model,
                self.push_loader,
            )
            pl_module.model.train()

            with open(
                os.path.join(
                    pl_module.model_dir,
                    f"push_log_epoch_{pl_module.current_epoch}.json",
                ),
                "w",
            ) as f:
                json.dump(log, f)


class SaveCallback(L.Callback):
    def __init__(
        self,
        monitor: str = "val_accuracy",
        mode: str = "max",
    ):
        self.monitor = monitor
        self.mode = mode
        match mode:
            case "max":
                self.best_metric = 0.0
            case "min":
                self.best_metric = float("inf")
            case _:
                raise ValueError(f"Mode {mode} not recognized")

        self.current_mode = None

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule):
        # reset the best metric if the mode has changed
        if self.current_mode != pl_module.model.train_mode:
            match self.mode:
                case "max":
                    self.best_metric = 0.0
                case "min":
                    self.best_metric = float("inf")
                case _:
                    raise ValueError(f"Mode {self.mode} not recognized")
            self.current_mode = pl_module.model.train_mode

        last_logs = trainer.logged_metrics
        metric = last_logs[self.monitor]

        save_flag = False
        match self.mode:
            case "max":
                if metric > self.best_metric:
                    save_flag = True
            case "min":
                if metric < self.best_metric:
                    save_flag = True
            case _:
                raise ValueError(f"Mode {self.mode} not recognized")

        if save_flag:
            self.best_metric = metric

            prefix = pl_module.model.train_mode
            # save the state dict
            torch.save(
                pl_module.model.state_dict(),
                os.path.join(
                    pl_module.model_dir, (f"best_{prefix}_model_state_dict.pth")
                ),
            )

            # save log
            logs = {}
            for key, value in last_logs.items():
                logs[key] = value.item()
            with open(
                os.path.join(pl_module.model_dir, f"best_{prefix}_model_log.json"), "w"
            ) as f:
                json.dump(logs, f)
