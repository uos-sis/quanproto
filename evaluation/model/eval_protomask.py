import quanproto.dataloader.complexity_segmentation as complexity_dataloader
import quanproto.dataloader.continuity_segmentation as continuity_dataloader
import quanproto.dataloader.contrastivity_segmentation as contrastivity_dataloader
import quanproto.dataloader.multi_label_segmentation as multi_label_dataloader
from eval_args import parser
from quanproto.dataloader.params import PIN_MEMORY
from quanproto.dataloader.segmentationmask import test_dataloader, validation_dataloader
from quanproto.utils.workspace import EXPERIMENTS_PATH

from quanproto.evaluation.config_parser import get_dataloader_fn_dict
from quanproto.evaluation.folder_utils import experiments_evaluation

parser.add_argument(
    "--segmentation_method",
    type=str,
    default="sam",
    choices=[
        "sam",
        "sam2",
        "slit",
    ],
    help="Name of the augmentation pipeline to be used for training",
)

parser.add_argument(
    "--fill_background_method",
    type=str,
    default="zero",
    choices=[
        "mean",
        "zero",
        "original",
    ],
)

parser.add_argument(
    "--min_bbox_size",
    type=float,
    default=0.0,
    help="Minimum size of the bounding box compared to the image size",
)

args = parser.parse_args()

condition = {"key": "segmentation_method", "value": args.segmentation_method}

# print the arguments
print(args)

TRAINING_PHASE = args.training_phase
EXPLANATION_TYPE = args.explanation_type

BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
CROP = args.crop
FILL_BACKGROUND_METHOD = args.fill_background_method
MIN_BBOX_SIZE = args.min_bbox_size

# only used for awa2 dataset in the contrastivity technique
MULTI_LABEL = args.multi_label
VALIDATION_SET = (
    args.validation_set
)  # INFO: If you want to use the validation set of the run instead of the test set
USE_BBOX = args.use_bbox  # INFO: If the model uses bounding boxes as input

experiment_config = {
    # INFO: All runs in the experiment will be evaluated so if you used multiple datasets you may
    # need to extend the experiment name to include the dataset name like PIPNet/cub200
    "experiment_dir": f"{EXPERIMENTS_PATH}/{args.experiment_sub_dir}",
    "dataset_dir": args.dataset_dir,
}

techniques = {
    "threshold": {},
    "general": {
        "metrics": [
            "accuracy",
            "top-3 accuracy",
            # "precision",
            # "recall",
            "f1",
            # "roc_auc",
        ]
    },
    "compactness": {
        "local_size_threshold": 0.1,
        "metrics": [
            "global size",
            "sparsity",
            "npr",
            # "local size"
        ],
    },
    "contrastivity": {
        "num_prototypes_per_sample": 5,
        "metrics": [
            "vlc",
            # "vac",
            # "plc",
            # "palc",
            "intra pd",  # INFO: only ProtoPNet, ProtoPool, ProtoTree
            # "intra fd",
            "inter pd",  # INFO: only ProtoPNet, ProtoPool, ProtoTree
            # "inter fd",
            # "entropy",
            # "histogram",
            # "projection",
        ],
        "use_bbox": USE_BBOX,  # used in vlc
    },
    "continuity": {
        "num_prototypes_per_sample": 5,
        "metrics": [
            # "vlc",
            # "vac",
            "plc",
            "psc",
            "prc",
            "palc",
            "pac",
            "cac",
            # "crc",
            # "stability",
        ],
        "use_bbox": USE_BBOX,  # used in vlc and stability
    },
    "complexity": {
        "num_prototypes_per_sample": 5,
        "metrics": [
            "ior",
            # "oirr",
            "object overlap",
            # "iou",
            "background overlap",
            "consistency",
        ],
        "use_bbox": USE_BBOX,  # used in consistency
    },
    "output_completeness": {
        "num_prototypes_per_sample": 5,
        "std": 0.05,
        "metrics": ["vlc", "vac", "plc", "psc", "prc", "palc", "pac"],
        "use_bbox": USE_BBOX,  # used in vlc
    },
    "topk_prototype_images": {
        "k": 5,
    },
}


def get_dataloader_fn_dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    crop=False,
    fill_background_method="zero",
    min_bbox_size=0.0,
    multi_label=False,
    validation_set=False,
):
    dataloder_to_use = validation_dataloader if validation_set else test_dataloader

    multi_label_dataloader_to_use = (
        multi_label_dataloader.validation_dataloader
        if validation_set
        else multi_label_dataloader.test_dataloader
    )

    contrastivity_dataloader_to_use = (
        contrastivity_dataloader.validation_dataloader
        if validation_set
        else contrastivity_dataloader.test_dataloader
    )

    continuity_dataloader_to_use = (
        continuity_dataloader.validation_dataloader
        if validation_set
        else continuity_dataloader.test_dataloader
    )

    complexity_dataloader_to_use = (
        complexity_dataloader.validation_dataloader
        if validation_set
        else complexity_dataloader.test_dataloader
    )

    dataloader_fn_dict = {
        "general": {
            "fn": dataloder_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
                "fill_background_method": fill_background_method,
                "min_bbox_size": min_bbox_size,
            },
        },
        "compactness": {
            "fn": dataloder_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
                "fill_background_method": fill_background_method,
                "min_bbox_size": min_bbox_size,
            },
        },
        "contrastivity": {
            "fn": contrastivity_dataloader_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
                "fill_background_method": fill_background_method,
                "min_bbox_size": min_bbox_size,
            },
        },
        "continuity": {
            "fn": continuity_dataloader_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
                "fill_background_method": fill_background_method,
                "min_bbox_size": min_bbox_size,
            },
        },
        "output_completeness": {
            "fn": contrastivity_dataloader_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
                "fill_background_method": fill_background_method,
                "min_bbox_size": min_bbox_size,
            },
        },
        "complexity": {
            "fn": complexity_dataloader_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
                "fill_background_method": fill_background_method,
                "min_bbox_size": min_bbox_size,
            },
        },
        "topk_prototype_images": {
            "fn": dataloder_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
                "fill_background_method": fill_background_method,
                "min_bbox_size": min_bbox_size,
            },
        },
    }

    return dataloader_fn_dict


# remove the techniques that are not selected
techniques = {k: v for k, v in techniques.items() if args.__dict__[k]}
print(techniques.keys())

if __name__ == "__main__":

    test_dataloader_fn_dict = get_dataloader_fn_dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        crop=CROP,
        fill_background_method=FILL_BACKGROUND_METHOD,
        min_bbox_size=MIN_BBOX_SIZE,
        multi_label=MULTI_LABEL,
        validation_set=VALIDATION_SET,
    )

    experiments_evaluation(
        experiment_config,
        techniques,
        training_phase=TRAINING_PHASE,
        explanation_type=EXPLANATION_TYPE,
        dataloader_fn_dict=test_dataloader_fn_dict,
        run_config_condition=condition,
    )
