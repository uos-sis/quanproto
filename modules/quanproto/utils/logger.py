import os
from typing import Any

from quanproto.logger import logger


def setup_logger(
    workspace_dir,
    project_name: str,
    config_all: dict[str, Any],
    additional_subdir: str = "",
) -> str:
    # region Logging -------------------------------------------------------------------------------
    logger.init(
        project=project_name,
        config=config_all,
        log_type=config_all["logger"],
        verbose=0,
        force_new=True,
    )
    run_name: str = logger.name()
    logger.info("Run name: {}".format(run_name))
    # endregion Logging ----------------------------------------------------------------------------

    config = logger.config()
    # region Log Folders ---------------------------------------------------------------------------
    aug_str = "_".join([key for key, value in config["augmentation_pipeline"]])
    range_str = "_".join([value for key, value in config["augmentation_pipeline"]])

    augmentation_subdir: str = (
        aug_str + "/" + range_str if aug_str != "none" else "none"
    )

    log_subdir: str = os.path.join(
        config["dataset"],
        augmentation_subdir,
        config["features"],
        "fold_{}".format(config["fold_idx"]),
    )

    if additional_subdir != "":
        log_subdir = os.path.join(log_subdir, additional_subdir)

    log_dir: str = os.path.join(
        workspace_dir, "experiments", logger.project(), log_subdir
    )
    os.makedirs(log_dir, exist_ok=True)
    logger.set_log_dir(log_dir)
    # endregion Log Folders ------------------------------------------------------------------------
    return log_dir
