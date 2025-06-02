import argparse

from quanproto.utils.workspace import DATASET_DIR

# region Input arguments ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Description of your program.")

# Add arguments
parser.add_argument(
    "--experiment_sub_dir",
    type=str,
    default="ProtoPNet/cub200",
)

parser.add_argument(
    "--dataset_dir",
    type=str,
    default=DATASET_DIR,
    help="Path to the dataset",
)

parser.add_argument(
    "--multi_label",
    action="store_true",
    help="Use multi-label classification",
)

parser.add_argument(
    "--validation_set",
    action="store_true",
    help="Use validation set",
)

parser.add_argument(
    "--use_bbox",
    action="store_true",
    help="Use bounding box",
)

parser.add_argument(
    "--training_phase",
    type=str,
    default="fine_tune",
)

parser.add_argument(
    "--explanation_type",
    type=str,
    default="upscale",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
)

parser.add_argument(
    "--crop",
    action="store_true",
)

parser.add_argument(
    "--general",
    action="store_true",
)

parser.add_argument(
    "--compactness",
    action="store_true",
)

parser.add_argument(
    "--contrastivity",
    action="store_true",
)

parser.add_argument(
    "--continuity",
    action="store_true",
)

parser.add_argument(
    "--complexity",
    action="store_true",
)

parser.add_argument(
    "--output_completeness",
    action="store_true",
)

parser.add_argument(
    "--topk_prototype_images",
    action="store_true",
)

parser.add_argument(
    "--threshold",
    action="store_true",
)
