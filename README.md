# Quanproto Python Library

## Paper Contributions
The repository contains the code for the following paper:
- **Schlinge et al. - 2025 - Comprehensive Evaluation of Prototype Neural Networks** 
  - [arXiv](https://arxiv.org/abs/2507.06819)
- **Meinert et al. - 2025 - ProtoMask: Segmentation-Guided Prototype Learning**



## Overview

Quanproto is a Python library designed to train and evaluate prototype networks,
featuring modules for model implementation, explanations, datasets, and
evaluation techniques.

# How to get started
To use the Quanproto library, you need to install the required packages. This 
can be done by running `pip install -r requirements.txt`. To run a script, you
can install the quanproto package or source the `.env` file.

## Dataset Preparation
The CUB200, Cars196, and AWA2 datasets are available through the
`datasets` module. To download the datasets, you just need to initialize the
dataset class. The code will then check if the dataset is available in the
specified directory and download it if it is not.

You can use the eda scripts in the evaluation directory to analyze the datasets.
If you run these scripts the first time, the datasets will be downloaded
automatically.

To split the dataset into training and validation sets, you can use the
split script in the evaluation directory. The script
will also download the dataset if needed.

The NICO dataset is not available for download through the `datasets` module.
You have to download it manually and place it in the specified directory.

1. Download the NICO dataset from dropbox: https://www.dropbox.com/scl/fo/ccng6n5ovl02x7f5nq5mq/AMfJAv-N7YCMXOgralg-_fM?rlkey=ol1twt2rlp3arqtf6fow83og6&e=2&dl=0
2. create a folder named `nico` in your dataset directory.
Important is that the folder is named `nico` (this is the default name in the implementation) and is located in the
dataset directory, which is specified with `DATASET_DIR` in the utils/workspace.py file.
3. create a `samples` folder in the `nico` folder and place the individual
animal or vehicle folders in the `samples` folder.
You can use this script to unzip all folders in a single folder:

```
for file in *.zip; do 
    dir="${file%.zip}"  # Remove .zip extension to get folder name
    mkdir -p "$dir"     # Create directory if it doesn't exist
    unzip "$file" -d "$dir" # Unzip into the created directory
done
```


The folder structure should look like
this:
```
nico
│   samples
│   ├── cow
│   │   ├── aside people
│   │   │   ├── 0001.jpg
│   │   │   ├── ...
│   │   ├── ...
```


## Training
After downloading the datasets, you can start training the models. The training
scripts are located in the `training` directory. The used model parameters can be found in the 
`modules/quanproto/models/protopnet/best_params.py` file.


## Evaluation

The evaluation scripts are located in the `evaluation/model` directory. Check out the `evaluation/model/eval_all_models.sh`
script for some examples on how to run the evaluation scripts.

You can also genrate latex tables for the results by running the `results2table.py` and the
`paper_table.py` scripts.
