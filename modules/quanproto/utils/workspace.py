import os

HOME_DIR = os.environ["HOME"]
DATASET_DIR = f"{HOME_DIR}/data/quanproto"
WORKSPACE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split("quanproto")[0], "quanproto"
)
EXPERIMENTS_PATH = os.path.join(WORKSPACE_PATH, "experiments")
