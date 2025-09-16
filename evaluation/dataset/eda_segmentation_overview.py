import os
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
from quanproto.utils.workspace import EXPERIMENTS_PATH

config = {
    "datasets": ["cub200", "dogs", "cars196"],
    "segmentation_methods": ["sam2", "slit"],
    "num_rows": 25,
}

# create a dataframe were all the statistics will be stored
global_stats = pd.DataFrame()
for dataset in config["datasets"]:
    for method in config["segmentation_methods"]:

        # load the markdown file with the statistics
        path = os.path.join(
            EXPERIMENTS_PATH,
            dataset,
            "all",
            "segmentation",
            f"{method}_overlap_distribution.md",
            # EXPERIMENTS_PATH,
            # dataset,
            # "all",
            # "segmentation",
            # f"{method}_size_distribution.md",
        )
        with open(path, "r") as f:
            markdown_table = f.read()

        # Replace pipes with commas and remove header lines
        csv_table = "\n".join(
            line for line in markdown_table.splitlines() if not line.startswith("|----")
        )
        csv_table = csv_table.replace("|", ",").replace(" ", "").strip()

        # Use StringIO to read the table into a pandas DataFrame
        data = StringIO(csv_table)
        df = pd.read_csv(data)
        df = df[["MeanMaskSize"]]

        # add the statistics to the global dataframe with the method as the column name
        identifier = f"{dataset}_{method}"
        global_stats[identifier] = df

# cut the dataframe to the number of rows specified in the config
global_stats = global_stats.head(config["num_rows"])

# create a matplotlib plot with the statistics
plt.figure(figsize=(10, 5))
plt.xlabel("Mask Index")
plt.ylabel("Mask Object Overlap")
# plt.ylabel("Mask Size / Image Size")
plt.title("Mean Object Overlap Distribution")
# plt.title("Mean Mask Size Distribution")

some_colors = [
    "#ff6384",
    "#36a2eb",
    "#ffce56",
    "#4bc0c0",
    "#9966ff",
    "#ff9f40",
]
# reverse the colors
some_colors.reverse()

# get all the columns and plot them
for column in global_stats.columns:
    plt.plot(
        global_stats[column],
        label=column,
        marker="o",
        alpha=0.7,
        color=some_colors.pop(),
    )

# plt.plot(global_stats["sam"], label="sam", marker="o", alpha=0.7, color="#ff6384")
# plt.plot(global_stats["slit"], label="slit", marker="o", alpha=0.7, color="#36a2eb")

plt.legend()
plt.xticks(range(config["num_rows"]))
plt.grid()


# save the plot
output_path = os.path.join(EXPERIMENTS_PATH, "mean_mask_overlap_distribution.pdf")
# output_path = os.path.join(EXPERIMENTS_PATH, "mean_mask_size_distribution.pdf")

plt.savefig(output_path)
