import quanproto.datasets.config_parser as quan_dataloader
from quanproto.utils.workspace import DATASET_DIR

config = {
    "dataset_dir": DATASET_DIR,
    "num_classes": 10,
    "dataset": "cub200",
    # "dataset": "dogs",
    # "dataset": "cars196",
    # "dataset": "nico",
    # "dataset": "awa2",
}

dataset = quan_dataloader.get_dataset(
    config["dataset_dir"], config["dataset"], dataset_name=config["dataset"] + "_mini"
)

# # TODO: not all datasets have implemented this method
dataset.make_mini_dataset(config["num_classes"])
dataset.save_mini_dataset()
