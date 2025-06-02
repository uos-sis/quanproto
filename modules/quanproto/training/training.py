import os
import sys
from typing import Any, Callable

import lightning as L
import torch
import torch.nn as nn
from optuna.integration import PyTorchLightningPruningCallback

from quanproto.logger.lightning_wrapper import LightningLogger


class LightningTrainerModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        compute_loss: Callable[[Any], tuple[dict[str, float], torch.Tensor]] = None,
        warm_optimizer: torch.optim.Optimizer | None = None,
        joint_optimizer: torch.optim.Optimizer | None = None,
        fine_tune_optimizer: torch.optim.Optimizer | None = None,
        warm_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        joint_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        fine_tune_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        warm_epochs: int = 10,
        joint_epochs: int = 10,
        fine_tune_epochs: int = 5,
        accumulate_grad_batches: int = 1,
        metric_callbacks: list[Callable] = [],
        model_dir: os.path = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])), "models"
        ),
    ):
        super().__init__()

        self.model = model
        self.compute_loss = compute_loss

        self.warm_epochs = warm_epochs
        self.joint_epochs = joint_epochs
        self.fine_tune_epochs = fine_tune_epochs

        self.warm_optimizer = warm_optimizer
        self.joint_optimizer = joint_optimizer
        self.fine_tune_optimizer = fine_tune_optimizer

        self.warm_scheduler = warm_scheduler
        self.joint_scheduler = joint_scheduler
        self.fine_tune_scheduler = fine_tune_scheduler

        self.metric_callbacks = metric_callbacks
        self.model_dir = model_dir

        self.total_epochs = warm_epochs + joint_epochs + fine_tune_epochs

        self.acc_grad_batches = accumulate_grad_batches
        self.log_on_step = False
        self.automatic_optimization = False

    def on_train_epoch_start(self) -> None:
        if self.current_epoch < self.warm_epochs:
            self.model.warmup()
        elif self.current_epoch < self.warm_epochs + self.joint_epochs:
            self.model.joint()
        elif (
            self.current_epoch
            < self.warm_epochs + self.joint_epochs + self.fine_tune_epochs
        ):
            self.model.fine_tune()
        else:
            raise ValueError("Epoch out of range")
        self.model.train()

    def training_step(self, batch, batch_index):
        # this is done because PIPNet uses a siamese approach
        if len(batch) == 2:
            inputs, target = batch
        if len(batch) == 3:
            inputs = torch.cat([batch[0], batch[1]])
            target = torch.cat([batch[2], batch[2]])

        with torch.enable_grad():
            output = self.model(inputs)
            log_loss, loss = self.compute_loss(self, output, target)

        for loss_key, loss_value in log_loss.items():
            loss_key = f"train_{loss_key}"
            self.log(loss_key, loss_value, on_step=False, on_epoch=True, sync_dist=True)

        # manual optimization
        if self.current_epoch < self.warm_epochs:
            opt = self.optimizers()[0]
        elif self.current_epoch < self.warm_epochs + self.joint_epochs:
            opt = self.optimizers()[1]
        elif (
            self.current_epoch
            < self.warm_epochs + self.joint_epochs + self.fine_tune_epochs
        ):
            opt = self.optimizers()[2]
        else:
            raise ValueError("Epoch out of range")

        # manual backward with accumulation
        loss /= self.acc_grad_batches
        self.manual_backward(loss)
        if (batch_index + 1) % self.acc_grad_batches == 0:
            opt.step()
            opt.zero_grad()

        for callback in self.metric_callbacks:
            callback.on_train_batch_end(self, output, target)

        del output, inputs, target

    def on_train_epoch_end(self):
        # step the schedule
        if self.current_epoch < self.warm_epochs:
            if self.warm_scheduler is not None:
                self.warm_scheduler.step()
                for i, param_group in enumerate(
                    self.optimizers()[0].optimizer.param_groups
                ):
                    self.log(
                        f"warm_lr_{i}",
                        param_group["lr"],
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

        elif self.current_epoch < self.warm_epochs + self.joint_epochs:
            if self.joint_scheduler is not None:
                self.joint_scheduler.step()
                for i, param_group in enumerate(
                    self.optimizers()[1].optimizer.param_groups
                ):
                    self.log(
                        f"joint_lr_{i}",
                        param_group["lr"],
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

        elif (
            self.current_epoch
            < self.warm_epochs + self.joint_epochs + self.fine_tune_epochs
        ):
            if self.fine_tune_scheduler is not None:
                self.fine_tune_scheduler.step()
                for i, param_group in enumerate(
                    self.optimizers()[2].optimizer.param_groups
                ):
                    self.log(
                        f"fine_tune_lr_{i}",
                        param_group["lr"],
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

    def on_validation_epoch_start(self):
        if self.current_epoch < self.warm_epochs:
            self.model.warmup()
        elif self.current_epoch < self.warm_epochs + self.joint_epochs:
            self.model.joint()
        elif (
            self.current_epoch
            < self.warm_epochs + self.joint_epochs + self.fine_tune_epochs
        ):
            self.model.fine_tune()
        else:
            raise ValueError("Epoch out of range")
        self.model.eval()

    def validation_step(self, batch, batch_index):
        # this is done because PIPNet uses a siamese approach
        if len(batch) == 2:
            inputs, target = batch
        if len(batch) == 3:
            inputs = torch.cat([batch[0], batch[1]])
            target = torch.cat([batch[2], batch[2]])

        with torch.no_grad():
            output = self.model(inputs)

            log_loss, _ = self.compute_loss(self, output, target)

        for loss_key, loss_value in log_loss.items():
            loss_key = f"val_{loss_key}"
            self.log(loss_key, loss_value, on_step=False, on_epoch=True, sync_dist=True)

        for callback in self.metric_callbacks:
            callback.on_validation_batch_end(self, output, target)

    def configure_optimizers(self):
        # INFO: return dummy optimizers to avoid error. These dummy optimizers should not be usable
        # if the epochs are correctly set
        return [
            (
                self.warm_optimizer
                if self.warm_optimizer is not None
                else torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
            ),
            (
                self.joint_optimizer
                if self.joint_optimizer is not None
                else torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
            ),
            (
                self.fine_tune_optimizer
                if self.fine_tune_optimizer is not None
                else torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
            ),
        ]


def train_lightning(
    model: LightningTrainerModel,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    callbacks: list[L.Callback] = [],
    eval_every_n_epochs: int = 1,
    no_progress_bar: bool = False,
) -> float:
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=LightningLogger(),
        enable_checkpointing=False,
        max_epochs=model.total_epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=eval_every_n_epochs,
        enable_progress_bar=not no_progress_bar,
        # strategy="ddp_find_unused_parameters_true",
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )

    return trainer.callback_metrics["val_accuracy"].item()


def tune_lightning(
    model: LightningTrainerModel,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    callbacks: list[L.Callback] = [],
) -> float:
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=LightningLogger(),
        enable_checkpointing=False,
        max_epochs=model.total_epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
        # strategy="ddp_find_unused_parameters_true",
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )

    for callback in callbacks:
        if isinstance(callback, PyTorchLightningPruningCallback):
            callback.check_pruned()

    return trainer.callback_metrics["val_accuracy"].item()
