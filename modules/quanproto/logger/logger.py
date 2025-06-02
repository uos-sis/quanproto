"""
This module contains the Logger class that creates the logger instance.
"""


class Logger:
    """
    A Singleton class that creates the logger instance.
    """

    _instance = None
    _verbose: bool = False

    @classmethod
    def verbose(cls):
        """
        Returns the verbose flag.
        """
        return cls._verbose

    @classmethod
    def get_instance(
        cls,
        project_name: str,
        config: dict | None = None,
        log_dir: str = "",
        log_type: str = "normal",
        verbose: bool = False,
        force_new: bool = True,
    ):
        """
        Returns the logger instance.
        """
        cls._verbose = verbose

        if force_new:
            del cls._instance
            cls._instance = None

        if cls._instance is None:
            if log_type == "tensorboard":
                from .tensorboard_logger import TensorBoardLogger

                cls._instance = TensorBoardLogger(project_name, config, log_dir)
            elif log_type == "normal":
                from .normal_logger import NormalLogger

                cls._instance = NormalLogger(project_name, config, log_dir)
        return cls._instance


def init(
    project: str,
    config: dict | None = None,
    log_dir: str = "",
    log_type: str = "normal",
    verbose: bool = False,
    force_new: bool = True,
):
    global _logger_instance
    _logger_instance = Logger.get_instance(
        project_name=project,
        config=config,
        log_dir=log_dir,
        log_type=log_type,
        verbose=verbose,
        force_new=force_new,
    )


def set_log_dir(log_dir: str):
    _logger_instance.set_log_dir(log_dir)


def log(message, step=None):
    _logger_instance.log(message, step)


def log_image(image, step=None, format="png"):
    _logger_instance.log_image(image, step, format)


def info(message: str) -> None:
    _logger_instance.console("--  INFO: " + message)


def debug(message: str) -> None:
    if Logger.verbose() == True:
        _logger_instance.console("-- DEBUG: " + message)


def project():
    return _logger_instance.project


def name():
    return _logger_instance.name


def config() -> dict:
    return _logger_instance.config


def log_dir():
    return _logger_instance.log_dir


def finish():
    _logger_instance.finish()
