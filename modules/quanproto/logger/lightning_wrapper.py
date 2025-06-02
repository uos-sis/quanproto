"""
This file contains the LightningLogger class which is a wrapper around the Lightning Logger class.
"""

from typing import Any

import lightning as L

from quanproto.logger import logger


class LightningLogger(L.pytorch.loggers.logger.Logger):
    """
    LightningLogger class is a wrapper around the Lightning Logger class.
    """

    def __init__(self):
        """
        Initializes the LightningLogger class.
        """
        super().__init__()

    def log_metrics(self, metrics: dict[str, float], step: int):
        """
        Log the metrics to the disk.
        """
        epoch = metrics["epoch"]
        del metrics["epoch"]
        logger.log(metrics, step=epoch)

    def log_hyperparams(self, params: dict[str, Any]):
        pass

    @property
    def version(self):
        """
        Returns the version of the logger.
        """
        return "0.0.1"

    @property
    def name(self):
        return logger.project

    def experiment(self):
        """
        Returns the run name
        """
        return logger.name

    @property
    def save_dir(self):
        """
        Returns the save directory.
        """
        return logger.log_dir
