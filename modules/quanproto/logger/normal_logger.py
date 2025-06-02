import json
import os
import random
import sys

from skimage import io


def generate_passphrase():
    # List of random words
    words = [
        "dauntless",
        "pretty",
        "forest",
        "banana",
        "sunset",
        "mountain",
        "ocean",
        "river",
        "garden",
        "elegant",
    ]

    # Selecting two random words
    word1 = random.choice(words)
    word2 = random.choice(words)

    # Generating a random 3-digit number
    number = random.randint(100, 999)

    # Concatenating the words and the number
    passphrase = f"{word1}-{word2}-{number}"

    return passphrase


class NormalLogger:
    """
    A simple logger that logs the metrics to the disk.
    """

    def __init__(
        self,
        project_name: str,
        entity: str | None = None,
        config: dict | None = None,
        log_dir: str = "",
    ):
        assert project_name is not None, "Project name cannot be None"
        assert project_name != "", "Project name cannot be empty"
        assert entity != "", "Entity cannot be empty"
        assert log_dir is not None, "Log directory cannot be None"

        self._project_name = project_name
        self._run_name = generate_passphrase()

        self._config = config
        self._logs = []
        self._console_log = []
        self._step = 0
        self.media_log_dir = ""

        if log_dir is not None:
            self.set_log_dir(log_dir)

    def set_log_dir(self, log_dir: str) -> None:
        """
        Set the log directory.
        """
        self._log_dir = log_dir
        # make sure the run name is unique
        while os.path.exists(
            os.path.join(self._log_dir, self._project_name, "logs", self._run_name)
        ):
            self._run_name = generate_passphrase()

        # create the log directory if it does not exist
        self.out_log_dir = os.path.join(
            self._log_dir, self._project_name, "logs", self._run_name
        )
        os.makedirs(self.out_log_dir, exist_ok=True)

        config_file = os.path.join(self.out_log_dir, f"{self._run_name}_config.json")
        with open(config_file, encoding="utf-8", mode="w") as f:
            json.dump(self._config, f)

    def log(self, metrics, step=None) -> None:
        """
        Log the metrics to the disk.
        """
        if step is not None:
            metrics = {"step": step, "metrics": metrics}
        self._logs.append(metrics)

    def console(self, message) -> None:
        """
        Log the message to the console.
        """
        self._console_log.append(message)
        print(message)

    def log_image(self, image, step=None, format="png") -> None:
        """
        Log the image to the disk.
        """
        if self._log_dir == "":
            self.set_log_dir(os.path.dirname(os.path.abspath(sys.argv[0])))

        if self.media_log_dir == "":
            self.media_log_dir = os.path.join(self.out_log_dir, "media")
            os.makedirs(self.media_log_dir, exist_ok=True)

        step = self._step if step is None else step

        image_file = os.path.join(
            self.media_log_dir, f"{self._run_name}_{step}.{format}"
        )
        io.imsave(image_file, image)

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
        if self._log_dir == "":
            self.set_log_dir(os.path.dirname(os.path.abspath(sys.argv[0])))

        log_file = os.path.join(self.out_log_dir, f"{self._run_name}_log.json")

        with open(log_file, encoding="utf-8", mode="w") as f:
            json.dump(self._logs, f)

        if len(self._console_log) != 0:
            console_log_file = os.path.join(
                self.out_log_dir,
                f"{self._run_name}_console_log.txt",
            )
            with open(console_log_file, encoding="utf-8", mode="w") as f:
                for log in self._console_log:
                    f.write(log + "\n")


if __name__ == "__main__":
    logger = NormalLogger(project_name="test_project", config={"test": 1})
    logger.log({"acc": 0.9, "loss": 0.1})
    logger.log({"acc": 0.8, "loss": 0.2})
    logger.finish()
