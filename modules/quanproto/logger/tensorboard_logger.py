"""
This file contains the WandbLogger class which is used to log the metrics and images to the wandb.
"""

import json
import os
import sys

from skimage import io
from torch.utils.tensorboard import SummaryWriter

from quanproto.logger.normal_logger import generate_passphrase


class TensorBoardLogger:
    """
    WandbLogger class is used to log the metrics and images to the wandb.
    """

    def __init__(
        self,
        project_name: str,
        config: dict | None = None,
        log_dir: str = "",
    ):
        assert project_name is not None, "Project name cannot be None"
        assert project_name != "", "Project name cannot be empty"

        self._project_name = project_name
        self._run_name = generate_passphrase()

        self._config = config
        self._console_log = []
        self._step = 0
        self._hparams = {}
        for key, value in config.items():
            if isinstance(value, (int, float, str)):
                self._hparams[key] = value

        if log_dir != "":
            self.set_log_dir(log_dir)

    def set_log_dir(self, log_dir: str) -> None:
        """
        Set the log directory.
        """
        self._log_dir = log_dir
        # make sure the run name is unique
        while os.path.exists(os.path.join(self._log_dir, "logs", self._run_name)):
            self._run_name = generate_passphrase()

        # create the log directory if it does not exist
        self.out_log_dir = os.path.join(self._log_dir, "logs", self._run_name)
        os.makedirs(self.out_log_dir, exist_ok=True)

        config_file = os.path.join(self.out_log_dir, f"{self._run_name}_config.json")
        with open(config_file, encoding="utf-8", mode="w") as f:
            json.dump(self._config, f)

        self.writer = SummaryWriter(self.out_log_dir)
        self.writer.add_text("config", json.dumps(self._config, indent=4))

    def log(self, metrics, step=None) -> None:
        """
        Log the metrics to the disk.
        """
        assert hasattr(
            self, "writer"
        ), "Writer is not initialized. Please call set_log_dir() first."

        if step is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        else:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value)

        if "val_accuracy" in metrics and step is not None:
            self.writer.add_hparams(
                self._hparams,
                metrics,
                run_name="hparam",
                global_step=step,
            )
        self.writer.flush()

    def console(self, message) -> None:
        """
        Log the message to the console.
        """
        self._console_log.append(message)
        print(message)

    @property
    def name(self):
        """
        Return the run name.
        """
        return self._run_name

    @property
    def project(self):
        """
        Return the project name.
        """
        return self._project_name

    @property
    def config(self):
        """
        Return the config dictionary.
        """
        return self._config

    @property
    def log_dir(self):
        """
        Return the log directory.
        """
        if self._log_dir == "":
            self.set_log_dir(os.path.dirname(os.path.abspath(sys.argv[0])))

        return self.out_log_dir

    def finish(self) -> None:
        """
        Save the logs to the disk.
        """
        self.writer.flush()
        self.writer.close()

        if self._log_dir == "":
            self.set_log_dir(os.path.dirname(os.path.abspath(sys.argv[0])))

        if len(self._console_log) != 0:
            console_log_file = os.path.join(
                self.out_log_dir,
                f"{self._run_name}_console_log.txt",
            )
            with open(console_log_file, encoding="utf-8", mode="w") as f:
                for log in self._console_log:
                    f.write(log + "\n")


if __name__ == "__main__":
    logger = TensorBoardLogger(
        project_name="test_project",
        config={"test": 1},
        log_dir="/home/pschlinge/repos/samxproto/tensorboard",
    )
    logger.log({"acc": 0.9, "loss": 0.1})
    logger.log({"acc": 0.8, "loss": 0.2})
    logger.finish()
