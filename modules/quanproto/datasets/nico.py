"""
Reference: https://www.dropbox.com/scl/fo/ccng6n5ovl02x7f5nq5mq/AMfJAv-N7YCMXOgralg-_fM?rlkey=ol1twt2rlp3arqtf6fow83og6&e=1&dl=0
"""

from quanproto.datasets.interfaces import DatasetBase
from quanproto.datasets import functional as F
import os
import time
import numpy as np
import multiprocessing
import shutil
import cv2
from PIL import Image


# We use the yellow, green and rest folder for training and the gray folder for testing
yellow_threshold = 15000
green_threshold = 15000
gray_threshold = 35000

# green Hue range 90-130, Saturation range 25-255, Brightness range 25-255
green_lower = np.array([90, 50, 50])
green_upper = np.array([130, 255, 255])

# yellow Hue range 40-90, Saturation range 50-255, Brightness range 50-255
yellow_lower = np.array([40, 50, 50])
yellow_upper = np.array([90, 255, 255])

# gray Hue range not important, Saturation range 0-100, Brightness range 100-255
gray_lower = np.array([0, 0, 100])
gray_upper = np.array([180, 100, 255])


def hsv_analyse(image_dir):
    # get the image file names
    all_files = []

    for root, directories, files in os.walk(image_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    # hsv_counts = []
    # with multiprocessing.Pool() as pool:
    #     hsv_counts = pool.map(hsv_mask, all_files)

    # use single process
    hsv_counts = []
    for image_path in all_files:
        hsv_counts.append(hsv_mask(image_path))

    # search for the files that are not valid and delete them
    pop_indices = []
    for i in range(len(all_files)):
        if hsv_counts[i] is None:
            pop_indices.append(i)

    # sort the indices in reverse order
    pop_indices.sort(reverse=True)
    if len(pop_indices) > 0:
        print(f"Found {len(pop_indices)} invalid files in {image_dir}")

    # delete the files
    for index in pop_indices:
        all_files.pop(index)
        hsv_counts.pop(index)
    assert len(all_files) == len(hsv_counts)

    return all_files, hsv_counts


def hsv_mask(image_path, image_size=(224, 224)):

    if (
        not image_path.endswith(".jpg")
        and not image_path.endswith(".jpeg")
        and not image_path.endswith(".png")
    ):
        # delete the file
        os.remove(image_path)
        return None

    try:
        image = Image.open(image_path)

        if image.format == "PNG":
            # UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
            if image.mode != "RGBA":
                image = image.convert("RGBA")
                image.save(image_path)

        # gray to rgb
        if image.mode == "L":
            image = image.convert("RGB")
            image.save(image_path)

        # rgba to rgb
        if image.mode == "RGBA":
            image = image.convert("RGB")
            image.save(image_path)

        # if the image is not in RGB mode, delete it
        if image.mode != "RGB":
            os.remove(image_path)
            return None

        # convert to numpy array
        image = np.asarray(image)

        # convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except:
        os.remove(image_path)
        return None

    # resize the image
    image = cv2.resize(image, image_size)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define masks for each category
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_gray = cv2.inRange(hsv, gray_lower, gray_upper)

    # Count the number of pixels for each category
    yellow_count = cv2.countNonZero(mask_yellow)
    green_count = cv2.countNonZero(mask_green)
    gray_count = cv2.countNonZero(mask_gray)

    return yellow_count, green_count, gray_count


class Nico(DatasetBase):
    def __init__(self, dataset_dir: str, dataset_name="nico") -> None:
        # first call the super class constructor with the dataset_directory and the dataset name
        super().__init__(dataset_dir, dataset_name)

        # region Special Variables ----------------------------------------------------------------
        # [sample_id, ...] (list of int)
        self._my_train_split = None
        self._my_test_split = None
        # endregion Special Variables --------------------------------------------------------------

        # region Special Functionalities -----------------------------------------------------------
        self._create_dataset_info()
        # endregion Special Functionalities --------------------------------------------------------

        # region Basic Functionalities -------------------------------------------------------------
        self._load_dataset_info()
        self._read_split_info()
        # endregion Basic Functionalities ----------------------------------------------------------

    def _delete_dataset_info(self) -> None:
        if os.path.isfile(os.path.join(self._root_dir, "class_names.txt")):
            os.remove(os.path.join(self._root_dir, "class_names.txt"))

        if os.path.isfile(os.path.join(self._root_dir, "sample_names.txt")):
            os.remove(os.path.join(self._root_dir, "sample_names.txt"))

        if os.path.isfile(os.path.join(self._root_dir, "sample_labels.txt")):
            os.remove(os.path.join(self._root_dir, "sample_labels.txt"))

        if os.path.isfile(os.path.join(self._root_dir, "train_test_split.txt")):
            os.remove(os.path.join(self._root_dir, "train_test_split.txt"))

        self.decompress_samples_folder()

        # check if split directory exists
        if os.path.exists(self._split_dir):
            shutil.rmtree(self._split_dir)

    def _create_dataset_info(self, force_new: bool = False) -> None:
        """
        Create the following files from the dataset directory:
        - class_names.txt
        - sample_names.txt
        - sample_labels.txt
        - train_test_split.txt
        """
        # region Check if the files exist ----------------------------------------------------------
        if os.path.isfile(os.path.join(self._root_dir, "sample_labels.txt")):
            if force_new:
                self._delete_dataset_info()
            else:
                return
        # endregion Check if the files exist -------------------------------------------------------

        # rename the folders if they contain a space
        for class_name in [
            f
            for f in os.listdir(self._sample_dir)
            if os.path.isdir(os.path.join(self._sample_dir, f))
        ]:
            if " " in class_name:
                os.rename(
                    os.path.join(self._sample_dir, class_name),
                    os.path.join(self._sample_dir, class_name.replace(" ", "_")),
                )

            for context in [
                f
                for f in os.listdir(os.path.join(self._sample_dir, class_name))
                if os.path.isdir(os.path.join(self._sample_dir, class_name, f))
            ]:
                if " " in context:
                    os.rename(
                        os.path.join(self._sample_dir, class_name, context),
                        os.path.join(self._sample_dir, class_name, context.replace(" ", "_")),
                    )

        # region hsv split -----------------------------------------------------------------------
        yellow_set, green_set, gray_set, rest_set = self._hsv_split()
        # endregion hsv split --------------------------------------------------------------------
        train_set = yellow_set + green_set + rest_set
        test_set = gray_set

        # this is the directory where we placed the dataset
        sample_names = {}
        sample_labels = {}
        class_names = {}

        # get a unique id for each sample
        sample_id = 0

        self._my_train_split = []
        self._my_test_split = []

        # region fetch all files -------------------------------------------------------------------
        class_folder = [
            f
            for f in os.listdir(self._sample_dir)
            if os.path.isdir(os.path.join(self._sample_dir, f))
        ]
        class_folder.sort()
        # walk through all the folders and subfolders
        for sample in train_set:
            # all samples are in a class folder like .../class/context/sample.jpg
            class_name = sample.split("/")[-3]
            context_name = sample.split("/")[-2]
            format_name = sample.split(".")[-1]
            new_name = (
                class_name
                + "/"
                + context_name
                + "/"
                + class_name
                + "_"
                + context_name
                + "_"
                + str(sample_id)
                + "."
                + format_name
            )
            # rename the file
            os.rename(sample, os.path.join(self._sample_dir, new_name))

            sample_names[sample_id] = new_name
            sample_labels[sample_id] = class_folder.index(class_name)
            self._my_train_split.append(sample_id)

            sample_id += 1

        for sample in test_set:
            # all samples are in a class folder like .../class/context/sample.jpg
            class_name = sample.split("/")[-3]
            context_name = sample.split("/")[-2]
            format_name = sample.split(".")[-1]
            new_name = (
                class_name
                + "/"
                + context_name
                + "/"
                + class_name
                + "_"
                + context_name
                + "_"
                + str(sample_id)
                + "."
                + format_name
            )
            # rename the file
            os.rename(sample, os.path.join(self._sample_dir, new_name))

            sample_names[sample_id] = new_name
            sample_labels[sample_id] = class_folder.index(class_name)
            self._my_test_split.append(sample_id)

            sample_id += 1
        # endregion fetch all files ----------------------------------------------------------------

        # region Create Dictionaries ---------------------------------------------------------------
        for index, class_name in enumerate(class_folder):
            class_names[index] = class_name
        # endregion Create Dictionaries ------------------------------------------------------------

        # region Create dataset files --------------------------------------------------------------
        # create the class_names.txt file
        with open(os.path.join(self._root_dir, "class_names.txt"), "w") as f:
            for class_id, class_name in class_names.items():
                f.write(f"{class_id} {class_name}\n")

        # create the sample_names.txt file
        with open(os.path.join(self._root_dir, "sample_names.txt"), "w") as f:
            for sample_id, sample_name in sample_names.items():
                f.write(f"{sample_id} {sample_name}\n")

        # create the sample_labels.txt file
        with open(os.path.join(self._root_dir, "sample_labels.txt"), "w") as f:
            for sample_id, sample_label in sample_labels.items():
                f.write(f"{sample_id} {sample_label}\n")

        # create the train_test_split.txt file
        with open(os.path.join(self._root_dir, "train_test_split.txt"), "w") as f:
            for sample_id in self._my_train_split:
                f.write(f"{sample_id} 1\n")
            for sample_id in self._my_test_split:
                f.write(f"{sample_id} 0\n")
        # endregion Create dataset files -----------------------------------------------------------

    def _load_dataset_info(self) -> None:
        super()._load_dataset_info()

        # load the test and train split
        self._my_train_split = []
        self._my_test_split = []
        with open(os.path.join(self._root_dir, "train_test_split.txt"), "r") as f:
            for line in f:
                sample_id, split = line.split()
                if split == "1":
                    self._my_train_split.append(int(sample_id))
                else:
                    self._my_test_split.append(int(sample_id))
        # make it into a numpy array
        self._my_train_split = np.array(self._my_train_split)
        self._my_test_split = np.array(self._my_test_split)
        self._test_split = self._my_test_split

    def _hsv_split(self):
        yellow_set = []
        green_set = []
        gray_set = []
        rest_set = []

        for class_name in [
            f
            for f in os.listdir(self._sample_dir)
            if os.path.isdir(os.path.join(self._sample_dir, f))
        ]:
            for context in [
                f
                for f in os.listdir(os.path.join(self._sample_dir, class_name))
                if os.path.isdir(os.path.join(self._sample_dir, class_name, f))
            ]:
                image_paths, hsv_counts = hsv_analyse(
                    os.path.join(self._sample_dir, class_name, context)
                )

                for i in range(len(image_paths)):
                    if hsv_counts[i] is None:
                        continue

                    if hsv_counts[i][2] > gray_threshold:
                        gray_set.append(image_paths[i])
                    elif hsv_counts[i][1] > green_threshold and hsv_counts[i][0] < yellow_threshold:
                        green_set.append(image_paths[i])
                    elif hsv_counts[i][0] > yellow_threshold and hsv_counts[i][1] < green_threshold:
                        yellow_set.append(image_paths[i])
                    else:
                        rest_set.append(image_paths[i])
        time.sleep(1)  # wait for invalid images to be deleted
        return yellow_set, green_set, gray_set, rest_set

    def split_dataset(
        self,
        k: int = 2,
        seed: int = 42,
        shuffle: bool = True,
        stratified: bool = True,
        train_size: float = 0.7,
    ) -> None:
        # Input checks
        if k < 1:
            raise ValueError("Number of folds must be greater than 0")
        if seed < 0:
            raise ValueError("Seed must be a positive integer")

        # value checks
        if len(self._sample_labels) == 0:
            raise ValueError("Sample labels are not set")
        if len(self._fold_splits) > 0:
            self.delete_split()
            self._test_split = self._my_test_split

        # self.decompress_samples_folder()
        # create a dictionary containing only the training samples
        subset_sample_labels = {
            sample_id: class_id
            for sample_id, class_id in self._sample_labels.items()
            if sample_id in self._my_train_split
        }

        self._fold_splits = F.k_fold_split(subset_sample_labels, k, seed, shuffle, stratified)

        self._num_folds = k
        self._is_balanced = [False] * self._num_folds

        self._make_split_dirs()
        self._save_split_info()
        self._copy_samples_to_split_dir()

        # update the folder sample names for each fold
        for fold_index in range(self._num_folds):
            fold_sample_ids = self._fold_splits[fold_index]
            train_sample_names = {
                sample_id: [
                    os.path.join(
                        self._sample_names[sample_id].split("/")[0],
                        self._sample_names[sample_id].split("/")[-1],
                    )
                ]
                for sample_id in fold_sample_ids[0]
            }
            validation_sample_names = {
                sample_id: [
                    os.path.join(
                        self._sample_names[sample_id].split("/")[0],
                        self._sample_names[sample_id].split("/")[-1],
                    )
                ]
                for sample_id in fold_sample_ids[1]
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
            sample_id: [
                os.path.join(
                    self._sample_names[sample_id].split("/")[0],
                    self._sample_names[sample_id].split("/")[-1],
                )
            ]
            for sample_id in self._test_split
        }
        self._folder_sample_names.update({"test": test_sample_names})

        # self.compress_samples_folder()

    def _read_split_info(self) -> None:
        super()._read_split_info()

        if not os.path.exists(self._split_dir):
            return

        self._change_folder_sample_names()

    def _change_folder_sample_names(self) -> None:
        """
        Change the sample names for the given folder
        """
        # update the folder sample names for each fold
        for fold_index in range(self._num_folds):
            for value in self._folder_sample_names[f"fold_{fold_index}"].values():
                for sample_id, sample_names in value.items():
                    for sample_name in sample_names:
                        class_name = sample_name.split("/")[0]
                        file_name = sample_name.split("/")[-1]
                        value[sample_id] = [os.path.join(class_name, file_name)]
        # update test set
        for sample_id, sample_names in self._folder_sample_names["test"].items():
            for sample_name in sample_names:
                class_name = sample_name.split("/")[0]
                file_name = sample_name.split("/")[-1]
                self._folder_sample_names["test"][sample_id] = [os.path.join(class_name, file_name)]
