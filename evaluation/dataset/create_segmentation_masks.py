import quanproto.datasets.config_parser as quan_dataloader
from quanproto.dataloader.custom import make_imageidx_dataloader
from quanproto.segmentation.config_parser import model_config_dic
from quanproto.segmentation.segmentation import generate_segment_masks
from quanproto.utils.workspace import DATASET_DIR

config = {
    "dataset_dir": DATASET_DIR,
    # "dataset": "cub200",
    # "dataset": "dogs",
    "dataset": "cars196",
    # "dataset": "nico",
    # "dataset": "awa2",
}

if __name__ == "__main__":

    dataset = quan_dataloader.get_dataset(config["dataset_dir"], config["dataset"])

    dataloader = make_imageidx_dataloader(config)
    generate_segment_masks(dataset, dataloader, model_config_dic["sam2"])
    generate_segment_masks(dataset, dataloader, model_config_dic["slit"])
