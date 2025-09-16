from quanproto.utils.workspace import EXPERIMENTS_PATH

from quanproto.evaluation.folder_utils import make_results_table

# best use dataset result folder level
FOLDER = f"{EXPERIMENTS_PATH}/ProtoMask/cub200"

experiment_config = {
    "experiment_dir": FOLDER,
    "out_dir": f"{FOLDER}/aggregate_results",
}

prefix = "sam"

if __name__ == "__main__":
    # For ProtoPNet, ProtoPool, PIPNet
    # make_results_table(experiment_config)
    # For ProtoMask
    make_results_table(experiment_config, prefix=prefix)
