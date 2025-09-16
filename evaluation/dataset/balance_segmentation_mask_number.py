import quanproto.datasets.config_parser as quan_dataloader
from quanproto.utils.workspace import DATASET_DIR

config = {
    "dataset_dir": DATASET_DIR,
    "dataset": {"name": "cub200", "sam2": 15, "slit": 8},
    "dataset": {"name": "dogs", "sam2": 15, "slit": 8},
    "dataset": {"name": "cars196", "sam2": 15, "slit": 8},
    "dataset": {"name": "nico", "sam2": 15, "slit": 8},
    "dataset": {"name": "awa2", "sam2": 15, "slit": 8},
}

if __name__ == "__main__":

    dataset = quan_dataloader.get_dataset(config["dataset_dir"], config["dataset"]["name"])

    dataset.reduce_masks("sam2", config["dataset"]["sam2"])
    dataset.balance_masks("sam2")

    dataset.reduce_masks("slit", config["dataset"]["slit"])
    dataset.balance_masks("slit")
