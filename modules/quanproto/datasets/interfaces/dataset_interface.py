import copy
import json
import multiprocessing
import os
import shutil
import subprocess

import numpy as np
import skimage as ski

import quanproto.datasets.functional as F


class DatasetBase:
    def __init__(self, dataset_dir, dataset_name, multi_label=False) -> None:
        # check if the dataset_dir exists, where the dataset will be placed
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)

        # Directories
        self._root_dir = os.path.join(dataset_dir, dataset_name)
        self._split_dir = os.path.join(self._root_dir, "train_test_splits")
        self._sample_dir = os.path.join(self._root_dir, "samples")
        self._segmentation_dir = os.path.join(self._root_dir, "segmentations")

        # Properties
        self._num_classes = None
        self._num_samples = None
        self._num_folds = None
        self._multi_label = multi_label

        # Dictionaries for the entire dataset
        # key: (sample_id) (int) value: (<class_name>/<sample_name>) (str)
        self._sample_names = {}
        # key: (sample_id) (int) value: (class_id) (int)
        self._sample_labels = {}
        # key: (class_id) (int) value: (<class_name>) (str)
        self._class_names = {}
        # key: (sample_id) (int) value: {dict with segmentation mask info}
        self._segmentation_info = {}
        self._segmentation_split_info = {}

        # Train and test splits
        # ([<sample_id>, ...], [<sample_id>, ...]) (tuple)
        self._fold_splits = []
        self._fold_splits_predefined = []

        self._class_weights = []
        self._num_train_samples = []
        # [<sample_id>, ...] (list)
        self._test_split = None
        self._test_split_predefined = None

        # dictionaries for individual folders
        self._is_balanced = []
        self._folder_sample_names = {}

    def multi_label(self) -> bool:
        return self._multi_label

    def has_splits(self) -> bool:
        return self._num_folds is not None

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise ValueError("Number of classes is not set")
        return self._num_classes

    def num_samples(self) -> int:
        if self._num_samples is None:
            raise ValueError("Number of samples is not set")
        return self._num_samples

    def num_folds(self) -> int:
        if self._num_folds is None:
            raise ValueError("Number of folds is not set")
        return self._num_folds

    def sample_dir(self) -> str:
        if self._sample_dir is None:
            raise ValueError("Sample directory is not set")
        return self._sample_dir

    def segmentation_dir(self) -> str:
        if self._segmentation_dir is None:
            raise ValueError("Segmentation directory is not set")
        return self._segmentation_dir

    def fold_info(self, k: int, dir_name) -> dict:
        if self._num_folds is None:
            raise ValueError("Folds have not been created")
        # check if the fold index is valid
        if k >= self._num_folds or k < 0:
            raise ValueError("Invalid fold index")
        # check if the directory name is valid
        if dir_name not in self._folder_sample_names[f"fold_{k}"]:
            raise ValueError("Invalid directory name")

        # get an array of sample ids
        sample_ids = []
        sample_paths = []
        sample_labels = []

        sample_masks = {}
        for id, paths in self._folder_sample_names[f"fold_{k}"][dir_name].items():
            for path in paths:
                sample_ids.append(id)
                sample_paths.append(path)
                sample_labels.append([self._sample_labels[id]])

                if len(self._segmentation_split_info) > 0:
                    # get the segmentation masks
                    for seg_method, masks_info in self._segmentation_split_info.items():
                        masks_paths = [
                            os.path.join(
                                ".".join(path.split(".")[:-1]),
                                masks_info[id]["masks"][mask_id],
                            )
                            for mask_id in masks_info[id]["masks"]
                        ]

                        if seg_method not in sample_masks:
                            sample_masks[seg_method] = {}
                            sample_masks[seg_method]["paths"] = []

                        sample_masks[seg_method]["paths"].append(masks_paths)

                        for prop_key, prop_dic in masks_info[id].items():
                            if prop_key == "masks":
                                continue
                            if prop_key not in sample_masks[seg_method]:
                                sample_masks[seg_method][prop_key] = []
                            sample_masks[seg_method][prop_key].append(
                                [val for val in prop_dic.values()]
                            )

        info = {
            "ids": sample_ids,
            "paths": sample_paths,
            "labels": sample_labels,
        }

        if len(sample_masks) > 0:
            info["masks"] = sample_masks
        return info

    def test_info(self) -> dict:
        if self._test_split is None:
            raise ValueError("Test split is not set")
        # get an array of sample ids
        sample_ids = []
        sample_paths = []
        sample_labels = []
        sample_masks = {}
        for id, paths in self._folder_sample_names["test"].items():
            for path in paths:
                sample_ids.append(id)
                sample_paths.append(path)
                sample_labels.append([self._sample_labels[id]])

                if len(self._segmentation_split_info) > 0:
                    # get the segmentation masks
                    for seg_method, masks_info in self._segmentation_split_info.items():
                        masks_paths = [
                            os.path.join(
                                ".".join(path.split(".")[:-1]),
                                masks_info[id]["masks"][mask_id],
                            )
                            for mask_id in masks_info[id]["masks"]
                        ]

                        if seg_method not in sample_masks:
                            sample_masks[seg_method] = {}
                            sample_masks[seg_method]["paths"] = []

                        sample_masks[seg_method]["paths"].append(masks_paths)

                        for prop_key, prop_dic in masks_info[id].items():
                            if prop_key == "masks":
                                continue
                            if prop_key not in sample_masks[seg_method]:
                                sample_masks[seg_method][prop_key] = []
                            sample_masks[seg_method][prop_key].append(
                                [val for val in prop_dic.values()]
                            )

        info = {
            "ids": sample_ids,
            "paths": sample_paths,
            "labels": sample_labels,
        }
        if len(sample_masks) > 0:
            info["masks"] = sample_masks
        return info

    def sample_info(self):
        sample_ids = []
        sample_paths = []
        sample_labels = []
        sample_masks = {}

        for id, paths in self._sample_names.items():
            sample_ids.append(id)
            sample_paths.append(paths)
            sample_labels.append([self._sample_labels[id]])

            if len(self._segmentation_info) > 0:
                # get the segmentation masks
                for seg_method, masks_info in self._segmentation_info.items():
                    masks_paths = [
                        os.path.join(
                            ".".join(paths.split(".")[:-1]),
                            masks_info[id]["masks"][mask_id],
                        )
                        for mask_id in masks_info[id]["masks"]
                    ]

                    if seg_method not in sample_masks:
                        sample_masks[seg_method] = {}
                        sample_masks[seg_method]["paths"] = []

                    sample_masks[seg_method]["paths"].append(masks_paths)

                    for prop_key, prop_dic in masks_info[id].items():
                        if prop_key == "masks":
                            continue
                        if prop_key not in sample_masks[seg_method]:
                            sample_masks[seg_method][prop_key] = []
                        sample_masks[seg_method][prop_key].append(
                            [val for val in prop_dic.values()]
                        )
        info = {
            "ids": sample_ids,
            "paths": sample_paths,
            "labels": sample_labels,
        }
        if len(sample_masks) > 0:
            info["masks"] = sample_masks
        return info

    # TODO: is this method necessary?
    def sample_names(self) -> list[str]:
        if len(self._sample_names) == 0:
            raise ValueError("Sample names are not set")
        return self._sample_names

    # TODO: is this method necessary?
    def sample_labels(self) -> dict:
        if len(self._sample_labels) == 0:
            raise ValueError("Sample labels are not set")
        return self._sample_labels

    def class_names(self) -> dict:
        if len(self._class_names) == 0:
            raise ValueError("Class names are not set")
        return self._class_names

    def class_weights(self, k: int) -> np.array:
        if self._num_folds is None:
            raise ValueError("Folds have not been created")
        if k >= self._num_folds or k < 0:
            raise ValueError("Invalid fold index")
        if len(self._class_weights) == 0:
            raise ValueError("Class weights are not set")
        return self._class_weights[k]

    def num_train_samples(self, k: int) -> int:
        if self._num_folds is None:
            raise ValueError("Folds have not been created")
        if k >= self._num_folds or k < 0:
            raise ValueError("Invalid fold index")
        if len(self._num_train_samples) == 0:
            raise ValueError("Number of train samples is not set")
        return self._num_train_samples[k]

    # TODO: is this method necessary?
    def fold_split(self, k: int) -> tuple[np.array, np.array]:
        if k >= self._num_folds or k < 0:
            raise ValueError("Invalid fold index")
        if len(self._fold_splits) == 0:
            raise ValueError("Fold splits are not set")
        return self._fold_splits[k]

    # TODO: is this method necessary?
    def test_split(self) -> np.array:
        if self._test_split is None:
            raise ValueError("Test split is not set")
        return self._test_split

    def save_mini_dataset(self) -> None:
        # delete if exists
        if os.path.exists(os.path.join(self._root_dir, "sample_names.txt")):
            os.remove(os.path.join(self._root_dir, "sample_names.txt"))

        with open(os.path.join(self._root_dir, "sample_names.txt"), "w") as f:
            for sample_id, name in self._sample_names.items():
                f.write(f"{sample_id} {name}\n")

        if os.path.exists(os.path.join(self._root_dir, "sample_labels.txt")):
            os.remove(os.path.join(self._root_dir, "sample_labels.txt"))

        with open(os.path.join(self._root_dir, "sample_labels.txt"), "w") as f:
            for sample_id, label in self._sample_labels.items():
                f.write(f"{sample_id} {label}\n")

        if os.path.exists(os.path.join(self._root_dir, "class_names.txt")):
            os.remove(os.path.join(self._root_dir, "class_names.txt"))

        with open(os.path.join(self._root_dir, "class_names.txt"), "w") as f:
            for class_id, name in self._class_names.items():
                f.write(f"{class_id} {name}\n")

        if len(self._segmentation_info) > 0:
            if os.path.exists(os.path.join(self._root_dir, "segmentation_info.json")):
                os.remove(os.path.join(self._root_dir, "segmentation_info.json"))

            with open(os.path.join(self._root_dir, "segmentation_info.json"), "w") as f:
                json.dump(self._segmentation_info, f)

        # save the predefined splits if they exist
        if len(self._fold_splits_predefined) > 0:
            if os.path.exists(os.path.join(self._root_dir, "train_test_split.txt")):
                os.remove(os.path.join(self._root_dir, "train_test_split.txt"))

            data_split = {}
            for fold_index in range(len(self._fold_splits_predefined)):
                for sample_id in self._fold_splits_predefined[fold_index][0]:
                    data_split[sample_id] = 1
                for sample_id in self._fold_splits_predefined[fold_index][1]:
                    data_split[sample_id] = 0
            for sample_id in self._test_split_predefined:
                data_split[sample_id] = 0

            # sort the keys
            data_split = dict(sorted(data_split.items()))

            with open(os.path.join(self._root_dir, "train_test_split.txt"), "w") as f:
                for sample_id, is_train in data_split.items():
                    f.write(f"{sample_id} {is_train}\n")

        # delete all class folders not in the class_names
        for folder in os.listdir(self._sample_dir):
            if folder not in self._class_names.values():
                shutil.rmtree(os.path.join(self._sample_dir, folder))

        # delete all class folders from the segmentation masks not in the class_names
        if len(self._segmentation_info) > 0:
            for seg_method in self._segmentation_info:
                for folder in os.listdir(os.path.join(self._segmentation_dir, seg_method)):
                    if folder not in self._class_names.values():
                        shutil.rmtree(os.path.join(self._segmentation_dir, seg_method, folder))

    def make_mini_dataset(self, n_classes: int = 10):
        # delete everything except the samples of the first n_classes

        # get image ids of the images that will be deleted
        delete_ids = []
        for class_id in range(n_classes, self._num_classes):
            del self._class_names[class_id]
            for sample_id, label in self._sample_labels.items():
                if label == class_id:
                    delete_ids.append(sample_id)

        # update dictionaries
        for sample_id in delete_ids:
            del self._sample_names[sample_id]
            del self._sample_labels[sample_id]
            if len(self._segmentation_info) > 0:
                for seg_method in self._segmentation_info:
                    del self._segmentation_info[seg_method][sample_id]

        self._num_classes = n_classes
        self._num_samples = len(self._sample_names)

        # delete ids from the predefined splits
        if len(self._fold_splits_predefined) > 0:
            for fold_index in range(len(self._fold_splits_predefined)):
                mask = ~np.isin(self._fold_splits_predefined[fold_index][0], delete_ids)
                new_train = self._fold_splits_predefined[fold_index][0][mask]
                mask = ~np.isin(self._fold_splits_predefined[fold_index][1], delete_ids)
                new_val = self._fold_splits_predefined[fold_index][1][mask]
                self._fold_splits_predefined[fold_index] = (new_train, new_val)

            mask = ~np.isin(self._test_split_predefined, delete_ids)
            self._test_split_predefined = self._test_split_predefined[mask]

        # delete the split if it exists
        self.delete_split()

    def _load_dataset_info(self) -> None:
        with open(os.path.join(self._root_dir, "sample_names.txt"), "r") as f:
            self._sample_names = {int(line.split()[0]): line.split()[1] for line in f}
        self._num_samples = len(self._sample_names)

        with open(os.path.join(self._root_dir, "sample_labels.txt"), "r") as f:
            self._sample_labels = {int(line.split()[0]): int(line.split()[1]) for line in f}

        with open(os.path.join(self._root_dir, "class_names.txt"), "r") as f:
            self._class_names = {int(line.split()[0]): line.split()[1] for line in f}
        self._num_classes = len(self._class_names)

        if os.path.exists(os.path.join(self._root_dir, "train_test_split.txt")):
            train_ids = []
            test_ids = []
            with open(os.path.join(self._root_dir, "train_test_split.txt"), "r") as f:
                for line in f:
                    sample_id, is_train = map(int, line.split())
                    if is_train == 1:
                        train_ids.append(sample_id)
                    else:
                        test_ids.append(sample_id)
            self._fold_splits_predefined = [(np.array(train_ids), np.array([]))]
            self._test_split_predefined = np.array(test_ids)

        # if segmentation masks are available, load the segmentation info
        if os.path.exists(os.path.join(self._root_dir, "segmentation_info.json")):
            with open(os.path.join(self._root_dir, "segmentation_info.json"), "r") as f:
                segmentation_info = json.load(f)

            # convert all index keys to int
            self._segmentation_info = {
                method: {
                    int(img_id): {
                        mask_prop_key: {
                            int(mask_id): prop_val for mask_id, prop_val in mask_prop_dict.items()
                        }
                        for mask_prop_key, mask_prop_dict in mask_info_dict.items()
                    }
                    for img_id, mask_info_dict in image_id_dict.items()
                }
                for method, image_id_dict in segmentation_info.items()
            }

        # if segmentation masks are available, load the segmentation info
        if os.path.exists(os.path.join(self._root_dir, "segmentation_split_info.json")):
            with open(os.path.join(self._root_dir, "segmentation_split_info.json"), "r") as f:
                segmentation_split_info = json.load(f)

            # convert all index keys to int
            self._segmentation_split_info = {
                method: {
                    int(img_id): {
                        mask_prop_key: {
                            int(mask_id): prop_val for mask_id, prop_val in mask_prop_dict.items()
                        }
                        for mask_prop_key, mask_prop_dict in mask_info_dict.items()
                    }
                    for img_id, mask_info_dict in image_id_dict.items()
                }
                for method, image_id_dict in segmentation_split_info.items()
            }

    def split_dataset(
        self,
        k: int = 2,
        seed: int = 42,
        shuffle: bool = True,
        stratified: bool = True,
        train_size: float = 0.7,
        predefined: bool = False,
    ) -> None:
        # Input checks
        if k < 1:
            raise ValueError("Number of folds must be greater than 0")
        if train_size <= 0 or train_size >= 1:
            raise ValueError("Train size must be between 0 and 1")
        if seed < 0:
            raise ValueError("Seed must be a positive integer")

        # value checks
        if len(self._sample_labels) == 0:
            raise ValueError("Sample labels are not set")
        if len(self._fold_splits) > 0 or self._test_split is not None:
            self.delete_split()

        if not predefined:
            self._fold_splits, self._test_split = F.create_splits(
                self._sample_labels, k, seed, shuffle, stratified, train_size
            )
            self._num_folds = k
            self._is_balanced = [False] * self._num_folds
        else:
            self._fold_splits = self._fold_splits_predefined
            self._test_split = self._test_split_predefined

            self._num_folds = len(self._fold_splits)
            self._is_balanced = [False] * self._num_folds

        self._make_split_dirs()
        self._save_split_info()
        self._copy_samples_to_split_dir()

        # update the folder sample names for each fold
        for fold_index in range(self._num_folds):
            fold_sample_ids = self._fold_splits[fold_index]
            train_sample_names = {
                sample_id: [self._sample_names[sample_id]] for sample_id in fold_sample_ids[0]
            }
            validation_sample_names = {
                sample_id: [self._sample_names[sample_id]] for sample_id in fold_sample_ids[1]
            }

            if f"fold_{fold_index}" not in self._folder_sample_names:
                self._folder_sample_names.update(
                    {f"fold_{fold_index}": {"train": train_sample_names}}
                )
                self._folder_sample_names[f"fold_{fold_index}"].update(
                    {"validation": validation_sample_names}
                )

        # update the folder sample names for the test set
        test_sample_names = {
            sample_id: [self._sample_names[sample_id]] for sample_id in self._test_split
        }
        self._folder_sample_names.update({"test": test_sample_names})

        # update the segmentation split info
        self._segmentation_split_info = copy.deepcopy(self._segmentation_info)
        self.save_segmentation_split_info()

        self._compute_class_weights()

    def _save_balanced(self, k) -> None:
        """
        Save the balanced train samples for folder k in balanced.txt
        """
        fold_names = []
        if self._fold_splits[k][0].size > 0:
            fold_names.append("train")
        if self._fold_splits[k][1].size > 0:
            fold_names.append("validation")

        for set_name in fold_names:
            with open(self._split_dir + f"/fold_{k}/balanced_" + set_name + ".txt", "w") as f:
                for sample_id, names in self._folder_sample_names[f"fold_{k}"][set_name].items():
                    f.write(f"{sample_id} {' '.join(names)}\n")

    def balance_classes(self) -> None:
        """
        Duplicate samples to balance the classes
        """
        if self._num_folds is None:
            raise ValueError("Folds have not been created")
        i = 0
        for k in range(self._num_folds):
            if self._is_balanced[k]:
                i += 1
        if i == self._num_folds:
            print("All folds are already balanced")
            return

        for k in range(self._num_folds):
            self._is_balanced[k] = True
            fold_names = []
            if self._fold_splits[k][0].size > 0:
                fold_names.append("train")
            if self._fold_splits[k][1].size > 0:
                fold_names.append("validation")

            for set_name in fold_names:
                fold_dir = os.path.join(self._split_dir, f"fold_{k}", set_name)

                fold_sample_names = {
                    sample_id: sample_names[0]
                    for sample_id, sample_names in self._folder_sample_names[f"fold_{k}"][
                        set_name
                    ].items()
                }
                fold_class_ids = {
                    sample_id: self._sample_labels[sample_id]
                    for sample_id in self._folder_sample_names[f"fold_{k}"][set_name].keys()
                }

                class_ids = np.array(list(fold_class_ids.values()))
                class_ids, counts = np.unique(class_ids, return_counts=True)
                max_count = np.max(counts)

                multiplier_dict = {}
                for label, count in zip(class_ids, counts):
                    multiplier, remainder = divmod(max_count, count)
                    multiplier_dict[label] = (multiplier - 1, remainder)

                all_args = []
                duplicates = {}
                seg_args = []
                for sample_id, sample_path in fold_sample_names.items():
                    label = fold_class_ids[sample_id]
                    multiplier, remainder = multiplier_dict[label]

                    new_sample_paths = [sample_path]
                    if multiplier > 0:

                        new_sample_paths.extend(
                            [
                                f"{'.'.join(sample_path.split('.')[:-1])}_{i}.{sample_path.split('.')[-1]}"
                                for i in range(multiplier)
                            ]
                        )
                        all_args.extend(
                            [
                                (
                                    os.path.join(fold_dir, sample_path),
                                    os.path.join(fold_dir, new_sample_paths[i + 1]),
                                )
                                for i in range(multiplier)
                            ]
                        )

                        # copy the segmentation mask folder for the image
                        if len(self._segmentation_info) > 0:
                            for seg_method in self._segmentation_info:
                                seg_args.extend(
                                    [
                                        (
                                            os.path.join(
                                                fold_dir + "_segmentations",
                                                seg_method,
                                                ".".join(sample_path.split(".")[:-1]),
                                            ),
                                            os.path.join(
                                                fold_dir + "_segmentations",
                                                seg_method,
                                                ".".join(new_sample_paths[i + 1].split(".")[:-1]),
                                            ),
                                        )
                                        for i in range(multiplier)
                                    ]
                                )

                    if remainder > 0:
                        new_sample_paths.append(
                            f"{'.'.join(sample_path.split('.')[:-1])}_{multiplier}.{sample_path.split('.')[-1]}"
                        )
                        all_args.append(
                            (
                                os.path.join(fold_dir, sample_path),
                                os.path.join(
                                    fold_dir,
                                    f"{'.'.join(sample_path.split('.')[:-1])}_{multiplier}.{sample_path.split('.')[-1]}",
                                ),
                            )
                        )
                        multiplier_dict[label] = (multiplier, remainder - 1)

                        # copy the segmentation mask folder for the image
                        if len(self._segmentation_info) > 0:
                            for seg_method in self._segmentation_info:
                                seg_args.append(
                                    (
                                        os.path.join(
                                            fold_dir + "_segmentations",
                                            seg_method,
                                            ".".join(sample_path.split(".")[:-1]),
                                        ),
                                        os.path.join(
                                            fold_dir + "_segmentations",
                                            seg_method,
                                            ".".join(new_sample_paths[-1].split(".")[:-1]),
                                        ),
                                    )
                                )

                    duplicates[sample_id] = new_sample_paths

                self._folder_sample_names[f"fold_{k}"][set_name] = duplicates

                # with multiprocessing.Pool(processes=8) as pool:
                #     pool.starmap(F.copy_sample, all_args)
                #     pool.starmap(shutil.copytree, seg_args)

                for arg in all_args:
                    F.copy_sample(*arg)
                for arg in seg_args:
                    shutil.copytree(*arg)

                self._save_balanced(k)

        self._compute_class_weights()

    def compress_samples_folder(self) -> None:
        self._compress_folder(self._sample_dir)
        for seg_method in self._segmentation_info:
            self._compress_folder(os.path.join(self._segmentation_dir, seg_method))

    def decompress_samples_folder(self) -> None:
        self._decompress_folder(self._sample_dir)
        for seg_method in self._segmentation_info:
            self._decompress_folder(os.path.join(self._segmentation_dir, seg_method))

    def _compress_folder(self, target_dir: str) -> None:
        for folder_name in os.listdir(target_dir):
            folder_path = os.path.join(target_dir, folder_name)
            if os.path.isdir(folder_path):
                tar_file = f"{folder_path}.tar.gz"
                subprocess.run(["tar", "-czf", tar_file, folder_name], cwd=target_dir)
                # Check if the archive was created successfully before removing the directory
                if os.path.exists(tar_file):
                    shutil.rmtree(folder_path)

    def _decompress_folder(self, target_dir: str) -> None:
        for file_name in os.listdir(target_dir):
            file_path = os.path.join(target_dir, file_name)
            if file_name.endswith(".tar.gz"):
                subprocess.run(["tar", "-xzf", file_name, "-C", target_dir], cwd=target_dir)
                # Check if the decompression was successful before removing the compressed file
                if os.path.exists(file_path[:-7]):  # Remove the ".tar.gz" extension
                    os.remove(file_path)

    def _copy_samples_to_split_dir(self) -> None:
        all_args = []

        # for all folds
        for fold_index in range(self._num_folds):
            # train samples
            all_args.extend(
                [
                    (
                        os.path.join(self._sample_dir, self._sample_names[sample_id]),
                        os.path.join(
                            self._split_dir,
                            f"fold_{fold_index}",
                            "train",
                            os.path.join(
                                self._class_names[self._sample_labels[sample_id]],
                                self._sample_names[sample_id].split("/")[-1],
                            ),
                        ),
                    )
                    for sample_id in self._fold_splits[fold_index][0]
                ]
            )

            seg_args = []
            for seg_method in self._segmentation_info:
                for image_id in self._fold_splits[fold_index][0]:
                    for mask_id in self._segmentation_info[seg_method][image_id]["masks"]:
                        seg_args.append(
                            (
                                os.path.join(
                                    self._segmentation_dir,
                                    seg_method,
                                    ".".join(self._sample_names[image_id].split(".")[:-1]),
                                    self._segmentation_info[seg_method][image_id]["masks"][mask_id],
                                ),
                                os.path.join(
                                    self._split_dir,
                                    f"fold_{fold_index}",
                                    "train_segmentations",
                                    seg_method,
                                    ".".join(self._sample_names[image_id].split(".")[:-1]),
                                    self._segmentation_info[seg_method][image_id]["masks"][mask_id],
                                ),
                            )
                        )
            all_args.extend(seg_args)

            # validation samples
            all_args.extend(
                [
                    (
                        os.path.join(self._sample_dir, self._sample_names[sample_id]),
                        os.path.join(
                            self._split_dir,
                            f"fold_{fold_index}",
                            "validation",
                            os.path.join(
                                self._class_names[self._sample_labels[sample_id]],
                                self._sample_names[sample_id].split("/")[-1],
                            ),
                        ),
                    )
                    for sample_id in self._fold_splits[fold_index][1]
                ]
            )

            seg_args = []
            for seg_method in self._segmentation_info:
                for image_id in self._fold_splits[fold_index][1]:
                    for mask_id in self._segmentation_info[seg_method][image_id]["masks"]:
                        seg_args.append(
                            (
                                os.path.join(
                                    self._segmentation_dir,
                                    seg_method,
                                    ".".join(self._sample_names[image_id].split(".")[:-1]),
                                    self._segmentation_info[seg_method][image_id]["masks"][mask_id],
                                ),
                                os.path.join(
                                    self._split_dir,
                                    f"fold_{fold_index}",
                                    "validation_segmentations",
                                    seg_method,
                                    ".".join(self._sample_names[image_id].split(".")[:-1]),
                                    self._segmentation_info[seg_method][image_id]["masks"][mask_id],
                                ),
                            )
                        )
            all_args.extend(seg_args)

        # test images
        all_args.extend(
            [
                (
                    os.path.join(self._sample_dir, self._sample_names[sample_id]),
                    os.path.join(
                        self._split_dir,
                        "test",
                        "test",
                        os.path.join(
                            self._class_names[self._sample_labels[sample_id]],
                            self._sample_names[sample_id].split("/")[-1],
                        ),
                    ),
                )
                for sample_id in self._test_split
            ]
        )

        seg_args = []
        for seg_method in self._segmentation_info:
            for image_id in self._test_split:
                for mask_id in self._segmentation_info[seg_method][image_id]["masks"]:
                    seg_args.append(
                        (
                            os.path.join(
                                self._segmentation_dir,
                                seg_method,
                                ".".join(self._sample_names[image_id].split(".")[:-1]),
                                self._segmentation_info[seg_method][image_id]["masks"][mask_id],
                            ),
                            os.path.join(
                                self._split_dir,
                                "test",
                                "segmentations",
                                seg_method,
                                ".".join(self._sample_names[image_id].split(".")[:-1]),
                                self._segmentation_info[seg_method][image_id]["masks"][mask_id],
                            ),
                        )
                    )
        all_args.extend(seg_args)

        # cropy samples in parallel
        # with multiprocessing.Pool(processes=8) as pool:
        #     pool.starmap(F.copy_sample, all_args)
        for arg in all_args:
            F.copy_sample(*arg)

        return None

    def _make_split_dirs(self) -> None:
        test_folders = []
        fold_folders = []
        if len(self._test_split) != 0:
            test_folders = ["test"]
        if len(self._fold_splits[0][0]) != 0:
            fold_folders.append("train")
        if len(self._fold_splits[0][1]) != 0:
            fold_folders.append("validation")

        # create the split folder
        if not os.path.isdir(self._split_dir):
            os.makedirs(self._split_dir)

        # create the test folder
        for folder_name in test_folders:
            os.makedirs(os.path.join(self._split_dir, "test", folder_name))
            # create the test/<class_name> folders
            for _, c_name in self._class_names.items():
                os.makedirs(os.path.join(self._split_dir, "test", folder_name, c_name))

        if len(self._test_split) != 0 and len(self._segmentation_info) != 0:
            for seg_method in self._segmentation_info:
                for img_id in self._test_split:
                    img_path = ".".join(self._sample_names[img_id].split(".")[:-1])
                    seg_dir = os.path.join(
                        self._split_dir, "test", "segmentations", seg_method, img_path
                    )
                    os.makedirs(seg_dir, exist_ok=True)

        # create the fold folders
        for fold_index in range(self._num_folds):
            os.makedirs(os.path.join(self._split_dir, f"fold_{fold_index}"))

            # create the fold/train and fold/validation folders
            for folder_name in fold_folders:
                os.makedirs(os.path.join(self._split_dir, f"fold_{fold_index}", folder_name))
                # create the fold/train/<class_name> and fold/validation/<class_name> folders
                for _, c_name in self._class_names.items():
                    os.makedirs(
                        os.path.join(self._split_dir, f"fold_{fold_index}", folder_name, c_name)
                    )

        if len(self._fold_splits[0][0]) != 0 and len(self._segmentation_info) != 0:
            for fold_index in range(self._num_folds):
                for seg_method in self._segmentation_info:
                    for img_id in self._fold_splits[fold_index][0]:
                        img_path = ".".join(self._sample_names[img_id].split(".")[:-1])
                        seg_dir = os.path.join(
                            self._split_dir,
                            f"fold_{fold_index}",
                            "train_segmentations",
                            seg_method,
                            img_path,
                        )
                        os.makedirs(seg_dir, exist_ok=True)

        if len(self._fold_splits[0][1]) != 0 and len(self._segmentation_info) != 0:
            for fold_index in range(self._num_folds):
                for seg_method in self._segmentation_info:
                    for img_id in self._fold_splits[fold_index][1]:
                        img_path = ".".join(self._sample_names[img_id].split(".")[:-1])
                        seg_dir = os.path.join(
                            self._split_dir,
                            f"fold_{fold_index}",
                            "validation_segmentations",
                            seg_method,
                            img_path,
                        )
                        os.makedirs(seg_dir, exist_ok=True)

        return None

    def _save_split_info(self) -> None:
        # save the train and validation split for each fold
        for fold_index, (train_indices, validation_indices) in enumerate(self._fold_splits):
            with open(
                os.path.join(
                    self._split_dir,
                    f"fold_{fold_index}",
                    f"split.txt",
                ),
                "w",
            ) as f:
                for sample_id in train_indices:
                    f.write(f"{sample_id} 1\n")
                for sample_id in validation_indices:
                    f.write(f"{sample_id} 0\n")

        # save the test set
        with open(os.path.join(self._split_dir, "test", "test.txt"), "w") as f:
            for sample_id in self._test_split:
                f.write(f"{sample_id}\n")

    def _read_split_info(self) -> None:
        if not os.path.exists(self._split_dir):
            return

        # get the number of folds
        self._num_folds = len(
            [
                name
                for name in os.listdir(self._split_dir)
                if os.path.isdir(os.path.join(self._split_dir, name)) and name.startswith("fold_")
            ]
        )
        self._is_balanced = [False] * self._num_folds

        # region Load fold_splits and test_split ---------------------------------------------------
        for fold_index in range(self._num_folds):
            train_indices = []
            validation_indices = []

            with open(
                os.path.join(
                    self._split_dir,
                    f"fold_{fold_index}",
                    f"split.txt",
                ),
                "r",
            ) as f:
                for line in f:
                    sample_id, is_train = line.split()
                    if is_train == "1":
                        train_indices.append(int(sample_id))
                    else:
                        validation_indices.append(int(sample_id))

            if len(train_indices) == 0:
                raise ValueError(f"Train split is empty for fold {fold_index}")
            self._fold_splits.append((np.array(train_indices), np.array(validation_indices)))

        # read the test set
        with open(os.path.join(self._split_dir, "test", "test.txt"), "r") as f:
            self._test_split = np.array([int(line) for line in f])
        # endregion Load fold_splits and test_split ------------------------------------------------

        # region Load folder_sample_names ----------------------------------------------------------
        for fold_index in range(self._num_folds):

            # read the train sample names
            sample_names = {
                id: [name]
                for id, name in self._sample_names.items()
                if id in self._fold_splits[fold_index][0]
            }
            if f"fold_{fold_index}" not in self._folder_sample_names:
                self._folder_sample_names.update({f"fold_{fold_index}": {"train": sample_names}})
            else:
                self._folder_sample_names[f"fold_{fold_index}"].update({"train": sample_names})

            # check if the train folder was balanced
            if os.path.exists(self._split_dir + f"/fold_{fold_index}/balanced_train.txt"):
                self._is_balanced[fold_index] = True
                # read the balanced train sample names
                with open(
                    os.path.join(
                        self._split_dir,
                        f"fold_{fold_index}",
                        f"balanced_train.txt",
                    ),
                    "r",
                ) as f:
                    sample_names = {}
                    for line in f:
                        sample_id, *names = line.split()
                        sample_names[int(sample_id)] = names

                    if f"fold_{fold_index}" not in self._folder_sample_names:
                        self._folder_sample_names.update(
                            {f"fold_{fold_index}": {"train": sample_names}}
                        )
                    else:
                        self._folder_sample_names[f"fold_{fold_index}"].update(
                            {"train": sample_names}
                        )

            # check if we have validation samples
            if self._fold_splits[fold_index][1].size > 0:
                # read the validation sample names
                sample_names = {
                    id: [name]
                    for id, name in self._sample_names.items()
                    if id in self._fold_splits[fold_index][1]
                }
                if f"fold_{fold_index}" not in self._folder_sample_names:
                    self._folder_sample_names.update(
                        {f"fold_{fold_index}": {"validation": sample_names}}
                    )
                else:
                    self._folder_sample_names[f"fold_{fold_index}"].update(
                        {"validation": sample_names}
                    )

                # check if the validation folder was balanced
                if os.path.exists(self._split_dir + f"/fold_{fold_index}/balanced_validation.txt"):
                    self._is_balanced[fold_index] = True
                    # read the balanced validation sample names
                    with open(
                        os.path.join(
                            self._split_dir,
                            f"fold_{fold_index}",
                            f"balanced_validation.txt",
                        ),
                        "r",
                    ) as f:
                        sample_names = {}
                        for line in f:
                            sample_id, *names = line.split()
                            sample_names[int(sample_id)] = names

                        if f"fold_{fold_index}" not in self._folder_sample_names:
                            self._folder_sample_names.update(
                                {f"fold_{fold_index}": {"validation": sample_names}}
                            )
                        else:
                            self._folder_sample_names[f"fold_{fold_index}"].update(
                                {"validation": sample_names}
                            )

        # read the folder sample names for the test set
        sample_names = {
            id: [name] for id, name in self._sample_names.items() if id in self._test_split
        }
        self._folder_sample_names["test"] = sample_names
        # endregion Load folder_sample_names -------------------------------------------------------
        self._compute_class_weights()

    def _compute_class_weights(self) -> None:
        for fold_index in range(self._num_folds):

            weights = []
            # compute the class weights for the train set
            fold_names = []
            if self._fold_splits[fold_index][0].size > 0:
                fold_names.append("train")
            if self._fold_splits[fold_index][1].size > 0:
                fold_names.append("validation")
            for folder_name in fold_names:

                if self._multi_label:
                    info = self.fold_info(fold_index, folder_name)
                    counts = np.sum(info["labels"], axis=0)
                    num_samples = len(info["labels"])
                    num_classes = len(counts)
                else:
                    info = self.fold_info(fold_index, folder_name)
                    _, counts = np.unique(info["labels"], return_counts=True)
                    num_samples = np.sum(counts)
                    num_classes = len(counts)

                # not balanced
                # INFO: works better but why
                # class_freq = counts / num_samples
                # weights.append(1 / class_freq)

                # balanced
                weights.append(num_samples / (num_classes * counts))

            self._class_weights.append(weights)

    def delete_split(self) -> None:
        self._fold_splits = []
        self._test_split = None
        self._num_folds = None
        self._folder_sample_names = {}

        self._segmentation_split_info = {}
        self._class_weights = []
        self._is_balanced = []

        if os.path.exists(os.path.join(self._root_dir, "segmentation_split_info.json")):
            os.remove(os.path.join(self._root_dir, "segmentation_split_info.json"))

        if os.path.exists(self._split_dir):
            shutil.rmtree(self._split_dir)

    def fold_dirs(self, k: int) -> dict:
        if k >= self._num_folds or k < 0:
            raise ValueError("Invalid fold index")

        fold_dir = os.path.join(self._split_dir, f"fold_{k}")
        if not os.path.exists(fold_dir):
            raise ValueError("Fold directory does not exist")

        dirs = {}
        # Iterate through subfolders within fold_dir
        for folder_name in os.listdir(fold_dir):
            folder_path = os.path.join(fold_dir, folder_name)

            # Check if it's a directory
            if os.path.isdir(folder_path):
                dirs[folder_name] = folder_path

        return dirs

    def test_dirs(self):
        # return os.path.join(self._split_dir, "test", "test")
        test_dir = os.path.join(self._split_dir, "test")
        if not os.path.exists(test_dir):
            raise ValueError("Test directory does not exist")

        dirs = {}
        # Iterate through subfolders within test_dir
        for folder_name in os.listdir(test_dir):
            folder_path = os.path.join(test_dir, folder_name)

            # Check if it's a directory
            if os.path.isdir(folder_path):
                dirs[folder_name] = folder_path

        return dirs

    def make_fold_dir(self, k: int, name: str) -> str:
        if k >= self._num_folds or k < 0:
            raise ValueError("Invalid fold index")

        fold_dir = os.path.join(self._split_dir, f"fold_{k}", name)
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        return fold_dir

    def make_test_dir(self, name: str) -> str:
        test_dir = os.path.join(self._split_dir, "test", name)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        return test_dir

    def delete_segmentation_masks(self, method_name: str) -> None:
        # delete the segmentation masks from disk
        if os.path.exists(os.path.join(self._segmentation_dir, method_name)):
            shutil.rmtree(os.path.join(self._segmentation_dir, method_name))

        # delete the segmentation masks from the segmentation_info dictionary
        if method_name in self._segmentation_info:
            del self._segmentation_info[method_name]

        # TODO: also delete the segmentation mask from the split folders

    def save_image_segmentation_masks(self, method_name: str, image_id: int, masks_dict: dict):
        """
        saves the masks to disk and updates the segmentation_info dictionary

        Args:
            method_name (str): method name like sam
            image_id (int): image id
            masks_dict (dict): segmentation masks and other info that should be saved
                example:
                    { masks: [mask_obj (ski.image)]
                      size: [mask_size (float)]
                      shift: [center_shift (tuple)]
                    }
        """

        # region check input arguments -------------------------------------------------------------
        # check if the masks_dict has the key masks
        if "masks" not in masks_dict:
            raise ValueError("masks_dict should have a key masks")

        # all list should have the same length
        for key in masks_dict:
            if len(masks_dict[key]) != len(masks_dict["masks"]):
                raise ValueError(f"all lists in masks_dict should have the same length")
        # endregion check input arguments ----------------------------------------------------------

        # region make the segmentation directory --------------------------------------------------
        sample_path = self._sample_names[image_id]

        # remove the file extension
        sample_path = ".".join(sample_path.split(".")[:-1])

        segmentation_dir = os.path.join(self._segmentation_dir, method_name, sample_path)

        os.makedirs(segmentation_dir, exist_ok=True)

        # endregion make the segmentation directory ------------------------------------------------

        # region save the segmentation masks to disk and update the dictionary ------------------------------------------------------
        masks_names_dic = {}
        for i, mask in enumerate(masks_dict["masks"]):
            mask_name = f"mask_{i}.png"
            mask_path = os.path.join(segmentation_dir, mask_name)
            try:
                ski.io.imsave(mask_path, mask, check_contrast=False)
            except:
                print(f"Could not save mask {mask_path} to disk")
                raise ValueError("Could not save mask to disk")

            masks_names_dic[i] = mask_name

        del masks_dict["masks"]

        additional_info = {}
        for key in masks_dict:
            for i, value in enumerate(masks_dict[key]):
                if key not in additional_info:
                    additional_info[key] = {}
                additional_info[key][i] = value

        new_dict = {"masks": masks_names_dic}
        new_dict.update(additional_info)

        if method_name not in self._segmentation_info:
            self._segmentation_info[method_name] = {image_id: new_dict}
        else:
            self._segmentation_info[method_name].update({image_id: new_dict})

    def save_segmentation_info(self):
        # if file exists, remove it
        if os.path.exists(os.path.join(self._root_dir, "segmentation_info.json")):
            os.remove(os.path.join(self._root_dir, "segmentation_info.json"))

        # save self._segmentation_info to disk as json
        with open(os.path.join(self._root_dir, "segmentation_info.json"), "w") as f:
            json.dump(self._segmentation_info, f)

    def save_segmentation_split_info(self):
        # if file exists, remove it
        if os.path.exists(os.path.join(self._root_dir, "segmentation_split_info.json")):
            os.remove(os.path.join(self._root_dir, "segmentation_split_info.json"))

        # save self._segmentation_info to disk as json
        with open(os.path.join(self._root_dir, "segmentation_split_info.json"), "w") as f:
            json.dump(self._segmentation_split_info, f)

    def reduce_masks(self, method, num: int):
        """
        Reduce the number of masks in the segmentation dir so every image has at most k masks
        """
        if self._num_folds is None:
            raise ValueError("Folds have not been created")

        # get the current state of the folder
        if method not in self._segmentation_split_info.keys():
            # the folder has the state after the split was made
            current_state_dict = self._segmentation_info[method]
        else:
            # we already reduced or balanced the masks of the method
            current_state_dict = self._segmentation_split_info[method]

        for k in range(self._num_folds):
            fold_names = []
            if self._fold_splits[k][0].size > 0:
                fold_names.append("train")
            if self._fold_splits[k][1].size > 0:
                fold_names.append("validation")

            for set_name in fold_names:
                fold_dir = os.path.join(
                    self._split_dir, f"fold_{k}", set_name + "_segmentations", method
                )

                all_args = []
                for img_id, sample_paths in self._folder_sample_names[f"fold_{k}"][
                    set_name
                ].items():

                    for mask_id, mask_name in current_state_dict[img_id]["masks"].items():

                        if mask_id >= num:
                            # go through all folders of an image (duplicated folders from balance_classes)
                            for sample_path in sample_paths:
                                all_args.append(
                                    os.path.join(
                                        fold_dir,
                                        ".".join(sample_path.split(".")[:-1]),
                                        mask_name,
                                    )
                                )

                # remove the masks from disk
                for arg in all_args:
                    os.remove(arg)

        # make the same for the test set
        fold_dir = os.path.join(self._split_dir, "test", "segmentations", method)
        all_args = []

        test_sample_paths = {img_id: [self._sample_names[img_id]] for img_id in self._test_split}

        for img_id, sample_paths in test_sample_paths.items():
            for mask_id, mask_name in current_state_dict[img_id]["masks"].items():
                if mask_id >= num:
                    for sample_path in sample_paths:
                        all_args.append(
                            os.path.join(
                                fold_dir,
                                ".".join(sample_path.split(".")[:-1]),
                                mask_name,
                            )
                        )

        # remove the masks from disk
        for arg in all_args:
            os.remove(arg)

        # this is the dict that will be updated and later copied to self._segmentation_split_info
        tmp_dict = copy.deepcopy(current_state_dict)

        # delete the masks from the segmentation info dict
        for img_id, mask_dict in current_state_dict.items():
            for mask_id in mask_dict["masks"]:
                if mask_id >= num:
                    # remove the mask from the segmentation info dict
                    for prop_key in current_state_dict[img_id]:
                        del tmp_dict[img_id][prop_key][mask_id]

        self._segmentation_split_info[method] = tmp_dict

        self.save_segmentation_split_info()

    def balance_masks(self, method):
        """
        balance the number of masks in the split dir
        """
        if self._num_folds is None:
            raise ValueError("Folds have not been created")

        # get the current state of the folder
        if method not in self._segmentation_split_info.keys():
            # the folder has the state after the split was made
            current_state_dict = self._segmentation_info[method]
        else:
            # we already reduced or balanced the masks
            current_state_dict = self._segmentation_split_info[method]

        # this is the dict that will be updated and later copied to self._segmentation_split_info
        tmp_dict = copy.deepcopy(current_state_dict)

        for k in range(self._num_folds):
            fold_names = []
            if self._fold_splits[k][0].size > 0:
                fold_names.append("train")
            if self._fold_splits[k][1].size > 0:
                fold_names.append("validation")

            for set_name in fold_names:
                fold_dir = os.path.join(
                    self._split_dir, f"fold_{k}", set_name + "_segmentations", method
                )

                # compute the number of masks for each image
                num_masks = {}
                for img_id in self._folder_sample_names[f"fold_{k}"][set_name].keys():
                    num_masks[img_id] = len(current_state_dict[img_id]["masks"])

                # get the maximum number of masks
                max_masks = max(num_masks.values())

                multiplier_dict = {}
                for img_id, num_mask in num_masks.items():
                    multiplier, remainder = divmod(max_masks, num_mask)
                    multiplier_dict[img_id] = (multiplier - 1, remainder)

                all_args = []
                for img_id, sample_paths in self._folder_sample_names[f"fold_{k}"][
                    set_name
                ].items():
                    multiplier, remainder = multiplier_dict[img_id]

                    for mask_id, mask_name in current_state_dict[img_id]["masks"].items():
                        if multiplier > 0:
                            # go through all folders of an image (duplicated folders from balance_classes)
                            for sample_path in sample_paths:
                                mask_folder_name = os.path.join(
                                    fold_dir, ".".join(sample_path.split(".")[:-1])
                                )

                                new_mask_names = [" "]
                                new_mask_names.extend(
                                    [
                                        f"{'.'.join(mask_name.split('.')[:-1])}_{i}.{mask_name.split('.')[-1]}"
                                        for i in range(multiplier)
                                    ]
                                )
                                all_args.extend(
                                    [
                                        (
                                            os.path.join(mask_folder_name, mask_name),
                                            os.path.join(mask_folder_name, new_mask_names[i + 1]),
                                        )
                                        for i in range(multiplier)
                                    ]
                                )

                            # update segmentation info dict
                            for i in range(multiplier):
                                tmp_dict[img_id]["masks"][num_masks[img_id] * (i + 1) + mask_id] = (
                                    new_mask_names[i + 1]
                                )
                                for prop_key in current_state_dict[img_id]:
                                    if prop_key != "masks":
                                        tmp_dict[img_id][prop_key][
                                            num_masks[img_id] * (i + 1) + mask_id
                                        ] = tmp_dict[img_id][prop_key][mask_id]

                        if remainder > 0:
                            for sample_path in sample_paths:
                                mask_folder_name = os.path.join(
                                    fold_dir, ".".join(sample_path.split(".")[:-1])
                                )

                                new_mask_name = f"{'.'.join(mask_name.split('.')[:-1])}_{multiplier}.{mask_name.split('.')[-1]}"
                                all_args.append(
                                    (
                                        os.path.join(mask_folder_name, mask_name),
                                        os.path.join(mask_folder_name, new_mask_name),
                                    )
                                )
                            # update segmentation info dict
                            tmp_dict[img_id]["masks"][
                                num_masks[img_id] * (multiplier + 1) + mask_id
                            ] = new_mask_name
                            for prop_key in current_state_dict[img_id]:
                                if prop_key != "masks":
                                    tmp_dict[img_id][prop_key][
                                        num_masks[img_id] * (multiplier + 1) + mask_id
                                    ] = tmp_dict[img_id][prop_key][mask_id]

                            remainder -= 1

                for arg in all_args:
                    F.copy_sample(*arg)

            # make the same for the test set
            fold_dir = os.path.join(self._split_dir, "test", "segmentations", method)

            test_sample_paths = {
                img_id: [self._sample_names[img_id]] for img_id in self._test_split
            }

            num_masks = {}
            for img_id in test_sample_paths.keys():
                num_masks[img_id] = len(current_state_dict[img_id]["masks"])

            # get the maximum number of masks
            max_masks = max(num_masks.values())

            multiplier_dict = {}
            for img_id, num_mask in num_masks.items():
                multiplier, remainder = divmod(max_masks, num_mask)
                multiplier_dict[img_id] = (multiplier - 1, remainder)

            all_args = []

            for img_id, sample_paths in test_sample_paths.items():
                multiplier, remainder = multiplier_dict[img_id]

                for mask_id, mask_name in current_state_dict[img_id]["masks"].items():
                    if multiplier > 0:
                        # go through all folders of an image (duplicated folders from balance_classes)
                        for sample_path in sample_paths:
                            mask_folder_name = os.path.join(
                                fold_dir, ".".join(sample_path.split(".")[:-1])
                            )

                            new_mask_names = [" "]
                            new_mask_names.extend(
                                [
                                    f"{'.'.join(mask_name.split('.')[:-1])}_{i}.{mask_name.split('.')[-1]}"
                                    for i in range(multiplier)
                                ]
                            )
                            all_args.extend(
                                [
                                    (
                                        os.path.join(mask_folder_name, mask_name),
                                        os.path.join(mask_folder_name, new_mask_names[i + 1]),
                                    )
                                    for i in range(multiplier)
                                ]
                            )

                        # update segmentation info dict
                        for i in range(multiplier):
                            tmp_dict[img_id]["masks"][num_masks[img_id] * (i + 1) + mask_id] = (
                                new_mask_names[i + 1]
                            )
                            for prop_key in current_state_dict[img_id]:
                                if prop_key != "masks":
                                    tmp_dict[img_id][prop_key][
                                        num_masks[img_id] * (i + 1) + mask_id
                                    ] = tmp_dict[img_id][prop_key][mask_id]

                    if remainder > 0:
                        for sample_path in sample_paths:
                            mask_folder_name = os.path.join(
                                fold_dir, ".".join(sample_path.split(".")[:-1])
                            )

                            new_mask_name = f"{'.'.join(mask_name.split('.')[:-1])}_{multiplier}.{mask_name.split('.')[-1]}"
                            all_args.append(
                                (
                                    os.path.join(mask_folder_name, mask_name),
                                    os.path.join(mask_folder_name, new_mask_name),
                                )
                            )
                        # update segmentation info dict
                        tmp_dict[img_id]["masks"][
                            num_masks[img_id] * (multiplier + 1) + mask_id
                        ] = new_mask_name
                        for prop_key in current_state_dict[img_id]:
                            if prop_key != "masks":
                                tmp_dict[img_id][prop_key][
                                    num_masks[img_id] * (multiplier + 1) + mask_id
                                ] = tmp_dict[img_id][prop_key][mask_id]

                        remainder -= 1

            for arg in all_args:
                F.copy_sample(*arg)

        self._segmentation_split_info[method] = tmp_dict

        self.save_segmentation_split_info()
