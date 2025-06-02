from eval_args import parser
from quanproto.evaluation.config_parser import get_dataloader_fn_dict
from quanproto.evaluation.folder_utils import experiments_evaluation
from quanproto.utils.workspace import EXPERIMENTS_PATH

args = parser.parse_args()

# print the arguments
print(args)

experiment_config = {
    # INFO: All runs in the experiment will be evaluated so if you used multiple datasets you may
    # need to extend the experiment name to include the dataset name like PIPNet/cub200
    "experiment_dir": f"{EXPERIMENTS_PATH}/{args.experiment_sub_dir}",
    "dataset_dir": args.dataset_dir,
}

# only used for awa2 dataset in the contrastivity technique
MULTI_LABEL = args.multi_label
VALIDATION_SET = (
    args.validation_set
)  # INFO: If you want to use the validation set of the run instead of the test set
USE_BBOX = args.use_bbox  # INFO: If the model uses bounding boxes as input

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
        "metrics": ["global size", "sparsity", "npr", "local size"],
    },
    "contrastivity": {
        "num_prototypes_per_sample": 5,
        "metrics": [
            # "vlc",
            # "vac",
            "plc",
            "palc",
            "intra pd",  # INFO: only ProtoPNet, ProtoPool, ProtoTree
            "intra fd",
            # "inter pd",  # INFO: only ProtoPNet, ProtoPool, ProtoTree
            # "inter fd",
            "entropy",
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
            "oirr",
            "object overlap",
            "iou",
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

# if not multilabel add inter distances to contrastivity
if not MULTI_LABEL:
    techniques["contrastivity"]["metrics"].extend(["inter pd"])
    techniques["contrastivity"]["metrics"].extend(["inter fd"])
    techniques["continuity"]["metrics"].extend(["crc"])

# remove the techniques that are not selected
techniques = {k: v for k, v in techniques.items() if args.__dict__[k]}
print(techniques.keys())

TRAINING_PHASE = args.training_phase
EXPLANATION_TYPE = (
    args.explanation_type
)  # INFO: "prp" if resnet was used as backbone, "upscale" as agnostic technique


BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
CROP = args.crop

if __name__ == "__main__":

    test_dataloader_fn_dict = get_dataloader_fn_dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        crop=CROP,
        multi_label=MULTI_LABEL,
        validation_set=VALIDATION_SET,
    )

    experiments_evaluation(
        experiment_config,
        techniques,
        training_phase=TRAINING_PHASE,
        explanation_type=EXPLANATION_TYPE,
        dataloader_fn_dict=test_dataloader_fn_dict,
    )
