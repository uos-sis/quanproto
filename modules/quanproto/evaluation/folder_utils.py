import io
import json
import os
from typing import Callable

import pandas as pd

import quanproto.evaluation.config_parser as config_parser
from quanproto.explanations.config_parser import load_model


def get_run_info(experiment_config: dict[str, str]) -> dict[str, dict[str, str]]:
    """This function uses the experiment_dir to find all training runs
    and returns a dictionary with the following structure:
    {
        run_name: {
            "config": path to the config file,
            "warmup": path to the best warmup model state_dict file,
            "joint": path to the best joint model state_dict file,
            "fine_tune": path to the best fine_tune model state_dict file,
        }
    }

    :param experiment_config: dictionary containing the key experiment_dir
    :type experiment_config: dict[str, str]
    :return: dictionary with information about the individual runs
    :rtype: dict[str, dict[str, str]]
    """

    root = experiment_config["experiment_dir"]

    run_info = {}

    # search the root directory recursively and search for the models and logs directory
    for root, dirs, files in os.walk(root):
        if "logs" in dirs and "models" in dirs:

            # get all runs from the logs directory
            log_runs = os.listdir(os.path.join(root, "logs"))
            # get all runs from the models directory
            model_runs = os.listdir(os.path.join(root, "models"))

            # get the intersection of the two runs
            runs = list(set(log_runs).intersection(set(model_runs)))

            # check if a run has a <folder_name>_config.json file if not remove it from the list
            rm_list = []
            for run in runs:
                if not os.path.exists(
                    os.path.join(root, "logs", run, f"{run}_config.json")
                ):
                    rm_list.append(run)

            # remove all runs that do not have a config file
            for run in rm_list:
                runs.remove(run)

            # check if a run has a state_dict.pth file if not remove it from the list
            rm_list = []
            for run in runs:
                find_state_dict = False
                for train_phase in ["warmup", "joint", "fine_tune"]:
                    if os.path.exists(
                        os.path.join(
                            root,
                            "models",
                            run,
                            f"best_{train_phase}_model_state_dict.pth",
                        )
                    ):
                        find_state_dict = True
                        break
                if not find_state_dict:
                    rm_list.append(run)

            # remove all runs that do not have a state_dict file
            for run in rm_list:
                runs.remove(run)

            # add the runs to the run_info dictionary
            for run in runs:
                run_info[run] = {
                    "config": os.path.join(root, "logs", run, f"{run}_config.json"),
                    "warmup": (
                        os.path.join(
                            root, "models", run, f"best_warmup_model_state_dict.pth"
                        )
                        if os.path.exists(
                            os.path.join(
                                root, "models", run, f"best_warmup_model_state_dict.pth"
                            )
                        )
                        else None
                    ),
                    "warmup_log": (
                        os.path.join(root, "models", run, f"best_warmup_model_log.json")
                        if os.path.exists(
                            os.path.join(
                                root, "models", run, f"best_warmup_model_log.json"
                            )
                        )
                        else None
                    ),
                    "joint": (
                        os.path.join(
                            root, "models", run, f"best_joint_model_state_dict.pth"
                        )
                        if os.path.exists(
                            os.path.join(
                                root, "models", run, f"best_joint_model_state_dict.pth"
                            )
                        )
                        else None
                    ),
                    "joint_log": (
                        os.path.join(root, "models", run, f"best_joint_model_log.json")
                        if os.path.exists(
                            os.path.join(
                                root, "models", run, f"best_joint_model_log.json"
                            )
                        )
                        else None
                    ),
                    "fine_tune": (
                        os.path.join(
                            root, "models", run, f"best_fine_tune_model_state_dict.pth"
                        )
                        if os.path.exists(
                            os.path.join(
                                root,
                                "models",
                                run,
                                f"best_fine_tune_model_state_dict.pth",
                            )
                        )
                        else None
                    ),
                    "fine_tune_log": (
                        os.path.join(
                            root, "models", run, f"best_fine_tune_model_log.json"
                        )
                        if os.path.exists(
                            os.path.join(
                                root, "models", run, f"best_fine_tune_model_log.json"
                            )
                        )
                        else None
                    ),
                }

    # error handling
    if len(run_info) == 0:
        raise FileNotFoundError("Could not find any runs in the experiment directory")
    return run_info


def get_technique_results(experiment_config):
    root = experiment_config["experiment_dir"]

    technique_info = {}

    # search the root directory recursively and search for the results directory
    for root, dirs, files in os.walk(root):
        if "results" in dirs:

            # get all runs from the results directory
            result_runs = os.listdir(os.path.join(root, "results"))

            for run in result_runs:
                run_dir = os.path.join(root, "results", run)

                # get all files in the run directory
                files = os.listdir(run_dir)

                # remove all files that are not json files
                files = [file for file in files if file.endswith(".json")]

                # a file name should be in the form of <run_name>_<technique>.json
                for file in files:
                    technique = "_".join(file.split("_")[1:]).split(".")[0]

                    if technique not in technique_info:
                        technique_info[technique] = {}
                    technique_info[technique][run] = os.path.join(run_dir, file)

    # error handling
    if len(technique_info) == 0:
        raise FileNotFoundError(
            "Could not find any results in the experiment directory"
        )
    return technique_info


def run_statistics(config):

    result_dicts = []
    for run_name, run_json in config.items():
        with open(run_json, "r") as f:
            result_dicts.append(json.load(f))

    # make a dataframe from the config
    keys = list(result_dicts[0].keys())

    result_df = pd.DataFrame(columns=keys)

    for result in result_dicts:
        values = list(result.values())
        result_df.loc[len(result_df)] = values

    result_df = result_df.describe()

    # if there are NaN values in the result_df, replace them with 0
    result_df = result_df.fillna(0)

    return result_df


def load_results_table(path):
    # check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find the results table")

    # check if the file is a md or tex file
    if path.endswith(".md"):
        with open(path, "r") as f:
            df = pd.read_table(
                f,
                sep="|",
                engine="python",
                skiprows=[1],
                header=0,
            )
            # skip first and last column
            df = df.iloc[:, 1:-1]
            # remove spaces from all entries in the dataframe
            df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
            df = df.set_index(df.columns[0])
    elif path.endswith(".tex"):
        file_string = ""
        with open(path, "r") as f:
            # skip all lines starting with \
            for line in f:
                if not line.startswith("\\"):
                    # del the \\ at the end of the line
                    line = line[:-3] + "\n"
                    file_string += line

        file_string = io.StringIO(file_string)
        df = pd.read_table(file_string, sep="&", engine="python")
        # remove spaces from all entries in the dataframe
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        df = df.set_index(df.columns[0])

    return df


def get_aggregate_results(experiment_config):
    """find the aggregate_results directory in the experiment directory and
    load all the results from the directory

    :param experiment_config: dictionary containing the experiment_dir
    :type experiment_config: dict[str, str]
    """

    root = experiment_config["experiment_dir"]
    root_folder = root.split("/")[-1]

    aggregate_results = {}

    # search the root directory recursively and search for the aggregate_results directory
    for root, dirs, _ in os.walk(root):
        if "aggregate_results" in dirs:

            # split the root directory to get the experiment name
            folder_names = root.split("/")
            # find the index of the root_folder in the folder_names
            idx = folder_names.index(root_folder)
            folder_names = folder_names[idx + 1 :]

            # get all technique files from the results directory
            technique_files = os.listdir(os.path.join(root, "aggregate_results"))

            # technique_dic = {}
            for technique_file in technique_files:
                technique = "_".join(technique_file.split("_")[:-1])

                if technique not in aggregate_results:
                    aggregate_results[technique] = {}
                # add the technique_dic to the aggregate_results dictionary
                dict_reference = aggregate_results[technique]
                for folder in folder_names:
                    if folder not in dict_reference:
                        dict_reference[folder] = {}
                    dict_reference = dict_reference[folder]

                if technique_file.endswith(".md"):
                    dict_reference["md"] = os.path.join(
                        root, "aggregate_results", technique_file
                    )
                elif technique_file.endswith(".tex"):
                    dict_reference["tex"] = os.path.join(
                        root, "aggregate_results", technique_file
                    )

    # error handling
    if len(aggregate_results) == 0:
        raise FileNotFoundError(
            "Could not find any results in the experiment directory"
        )
    return aggregate_results


def make_results_table(experiment_config):

    technique_info = get_technique_results(experiment_config)

    for technique, run_info in technique_info.items():
        stats_df = run_statistics(run_info)
        save_statistics(stats_df, experiment_config, technique)


def save_statistics(stats_df, experiment_config, technique):

    out_dir = experiment_config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # crop to .3f
    stats_df = stats_df.map(lambda x: f"{x:.2f}")
    markdown_table = stats_df.to_markdown()

    # get the indices of the dataframe
    index = stats_df.index
    # check if a % is in the index and if so switch it with \%
    index = [idx.replace("%", r"\%") for idx in index]
    stats_df.index = index

    latex_table = stats_df.to_latex()

    if os.path.exists(os.path.join(out_dir, f"{technique}_results.md")):
        os.remove(os.path.join(out_dir, f"{technique}_results.md"))
    if os.path.exists(os.path.join(out_dir, f"{technique}_results.tex")):
        os.remove(os.path.join(out_dir, f"{technique}_results.tex"))

    with open(os.path.join(out_dir, f"{technique}_results.md"), "w") as f:
        f.write(markdown_table)
    with open(os.path.join(out_dir, f"{technique}_results.tex"), "w") as f:
        f.write(latex_table)


def experiments_evaluation(
    experiment_config: dict[str, str],
    techniques: dict,
    training_phase: str = "fine_tune",
    explanation_type="upscale",
    dataloader_fn_dict=config_parser.get_dataloader_fn_dict(),
):
    """This function goes through all runs in the experiment directory and evaluates the
    model with the given technique. The results are saved in the run directory under the
    subdirectory results.

    :param experiment_config: dictionary containing the experiment_dir and dataset_dir
    :type experiment_config: dict[str, str]
    :param technique: list of technique names to evaluate
    :type technique: list[str]
    :param training_phase: prefix used to identify the state dict that is loaded, defaults to "fine_tune"
    :type training_phase: str, optional
    :param explanation_type: method used to generate explanations in the input space, defaults to "upscale"
    :type explanation_type: str, optional
    :param dataloader_fn_dict: dictionary containing dataloader functions for the corresponding techniques, defaults to config_parser.default_dataloader_fn_dict
    :type dataloader_fn_dict: dict[str, Callable], optional
    """

    # TODO: there is no error handling for the training_phase or explanation_type parameters

    run_info = get_run_info(experiment_config)

    # go through all runs
    for run_name, run_dict in run_info.items():

        # add the dataset_dir from the experiment_config to the run_info
        run_dict["dataset_dir"] = experiment_config["dataset_dir"]

        # go through all techniques
        for technique, args in techniques.items():
            # get the dataloader function for the technique
            dataloader_fn = dataloader_fn_dict[technique]
            technique_fn = config_parser.evaluation_techniques_fn[technique]

            # run the evaluation
            print(f"Run: {run_name}")
            history = run_evaluation(
                run_dict,
                technique_fn,
                args,
                dataloader_fn,
                explanation_type,
                training_phase,
            )

            result_dir = os.path.dirname(run_dict["config"]).replace("logs", "results")
            os.makedirs(result_dir, exist_ok=True)

            # save metric as json file in the log dir from the current run
            with open(
                os.path.join(result_dir, f"{run_name}_{technique}.json"),
                "w",
            ) as f:
                json.dump(history, f)


def run_evaluation(
    run_info: dict[str, str],
    evaluation_fn: Callable,
    evaluation_fn_args: dict,
    dataloader_fn: Callable,
    explanation_type="upscale",
    training_phase: str = "fine_tune",
) -> None:

    # load the config file
    with open(run_info["config"], "r") as f:
        run_config = json.load(f)

    # check if the needed information to make a dataloader is in the config file
    if "dataset" not in run_config:
        raise KeyError("Could not find dataset in the config file")
    if "fold_idx" not in run_config:
        raise KeyError("Could not find fold_idx in the config file")

    run_config["dataset_dir"] = run_info["dataset_dir"]
    dataloader = dataloader_fn["fn"](run_config, **dataloader_fn["args"])

    model = load_model(
        run_config,
        explanation_type,
        run_info[training_phase],
    )

    with open(run_info[f"{training_phase}_log"], "r") as f:
        run_log = json.load(f)

    if "val_threshold" in run_log:
        model.multi_label_threshold = run_log[
            "val_threshold"
        ]  # we could also use the train_threshold

    model.cuda()
    history = evaluation_fn(model, dataloader, **evaluation_fn_args)

    # delete the model to free up memory
    del model

    return history
