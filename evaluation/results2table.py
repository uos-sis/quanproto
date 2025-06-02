from quanproto.utils.workspace import EXPERIMENTS_PATH

from quanproto.evaluation.folder_utils import make_results_table

# best use dataset result folder level
FOLDER = f"{EXPERIMENTS_PATH}/ProtoPNet/awa2"

experiment_config = {
    "experiment_dir": FOLDER,
    "out_dir": f"{FOLDER}/aggregate_results",
}

if __name__ == "__main__":
    make_results_table(experiment_config)
