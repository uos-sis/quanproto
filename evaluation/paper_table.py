import os

import pandas as pd
from quanproto.utils.workspace import EXPERIMENTS_PATH

from quanproto.evaluation.folder_utils import get_aggregate_results, load_results_table

FOLDER = f"{EXPERIMENTS_PATH}/PIPNet"

experiment_config = {
    "experiment_dir": FOLDER,
    "out_dir": f"{FOLDER}/aggregate_results",
}

if __name__ == "__main__":
    aggregated_results = get_aggregate_results(experiment_config)

    technique_df_list = []
    for technique, sub_dict in aggregated_results.items():

        technique_df = pd.DataFrame()

        # the sub_dict contains the datasets
        datsets = list(sub_dict.keys())

        for dataset in datsets:
            path = sub_dict[dataset]["tex"]
            df = load_results_table(path)

            mean_row = df.loc["mean"]
            std_row = df.loc["std"]

            # Create a new DataFrame with the "mean ± std" format
            formatted_data = {
                col: f"{mean_row[col]:.2f} $\\pm$ {std_row[col]:.2f}" for col in df.columns
            }
            df = pd.DataFrame(formatted_data, index=[dataset])
            technique_df = pd.concat([technique_df, df])

        technique_df_list.append(technique_df)

    for technique, df in zip(aggregated_results.keys(), technique_df_list):
        out_dir = experiment_config["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        latex_table = df.to_latex()

        if os.path.exists(os.path.join(out_dir, f"{technique}_results.tex")):
            os.remove(os.path.join(out_dir, f"{technique}_results.tex"))

        with open(os.path.join(out_dir, f"{technique}_results.tex"), "w") as f:
            f.write(latex_table)
