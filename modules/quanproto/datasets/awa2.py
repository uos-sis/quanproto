import os

import pandas as pd

ANIMALS_WITH_ATTRIBUTES_2_URL = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"
from quanproto.datasets.interfaces.dataset_interface import DatasetBase

# visual predicates that are used in the dataset
# <name> <num classes that have the attribute>

visual_predicates_dict = {
    0: 1,  # black 31
    1: 1,  # white 23
    2: 1,  # blue 4
    3: 1,  # brown 33
    4: 1,  # gray 26
    5: 1,  # orange 4
    6: 1,  # red 1
    7: 1,  # yellow 4
    8: 1,  # patches 15
    9: 1,  # spots 12
    10: 1,  # stripes 4
    11: 1,  # furry 39
    12: 1,  # hairless 13
    13: 1,  # toughskin 23
    14: 1,  # big 31
    15: 1,  # small 21
    16: 1,  # bulbous 26
    17: 1,  # lean 25
    18: 1,  # flippers 7
    19: 1,  # hands 3
    20: 1,  # hooves 12
    21: 1,  # pads 15
    22: 1,  # paws 27
    23: 1,  # longleg 14
    24: 1,  # longneck 5
    25: 1,  # tail 39
    26: 1,  # chewteeth 39
    27: 1,  # meatteeth 23
    28: 1,  # buckteeth 12
    29: 1,  # strainteeth 6
    30: 1,  # horns 8
    31: 1,  # claws 22
    32: 1,  # tusks 3
    33: 0,  # smelly 24
    34: 0,  # flys 1
    35: 0,  # hops 3
    36: 0,  # swims 10
    37: 0,  # tunnels 5
    38: 0,  # walks 40
    39: 0,  # fast 42
    40: 0,  # slow 20
    41: 0,  # strong 34
    42: 0,  # weak 10
    43: 0,  # muscle 29
    44: 1,  # bipedal 8
    45: 1,  # quadrapedal 43
    46: 0,  # active 37
    47: 0,  # inactive 23
    48: 0,  # nocturnal 15
    49: 0,  # hibernate 13
    50: 0,  # agility 33
    51: 0,  # fish 17
    52: 0,  # meat 20
    53: 0,  # plankton 3
    54: 0,  # vegetation 26
    55: 0,  # insects 4
    56: 0,  # forager 26
    57: 0,  # grazer 17
    58: 0,  # hunter 17
    59: 0,  # scavenger 6
    60: 0,  # skimmer 2
    61: 0,  # stalker 10
    62: 0,  # newworld 41
    63: 0,  # oldworld 44
    64: 1,  # arctic 9
    65: 1,  # coastal 8
    66: 1,  # desert 1
    67: 1,  # bush 11
    68: 1,  # plains 20
    69: 1,  # forest 22
    70: 1,  # fields 19
    71: 1,  # jungle 11
    72: 1,  # mountains 12
    73: 1,  # ocean 8
    74: 1,  # ground 41
    75: 1,  # water 10
    76: 1,  # tree 9
    77: 1,  # cave 4
    78: 0,  # fierce 21
    79: 0,  # timid 33
    80: 0,  # smart 34
    81: 0,  # group 30
    82: 0,  # solitary 28
    83: 0,  # nestspot 20
    84: 0,  # domestic 18
}


class AwA2(DatasetBase):
    def __init__(self, dataset_dir: str, dataset_name="awa2", multi_label: bool = True):
        super().__init__(dataset_dir, dataset_name, multi_label=multi_label)

        # Extra Dictionaries
        # key: class_id (int) value: binary attribute vector
        self._class_vis_predicates = {}
        # key: predicate_id (int) value: predicate_name (str)
        self._vis_predicate_names = {}

        self._vis_pred_ids = []
        self._num_vis_predicates = 0

        self._train_classes = {}
        self._test_classes = {}

        self._predicate_weights = []

        self._download_dataset(dataset_dir)
        self._load_dataset_info()

        # load the splits if they already exist
        self._read_split_info()

    def class_predicates(self) -> dict:
        return self._class_vis_predicates

    def num_classes(self) -> int:
        # count the 1s in the visual predicates
        num = sum(visual_predicates_dict.values())
        return num

    def predicate_names(self) -> dict:
        return self._vis_predicate_names

    def _download_dataset(self, dataset_dir) -> None:
        # check if the dataset_dir contains the CUB200-2011 dataset if not download and extract it
        if not os.path.isdir(self._root_dir):

            # check if the zip is already downloaded
            if not os.path.isfile(os.path.join(dataset_dir, "AwA2-data.zip")):
                # download the dataset
                os.system(f"wget {ANIMALS_WITH_ATTRIBUTES_2_URL} -P {dataset_dir}")

            # extract the dataset
            if ANIMALS_WITH_ATTRIBUTES_2_URL.split("/")[-1].split(".")[-1] == "zip":
                file_name = ANIMALS_WITH_ATTRIBUTES_2_URL.split("/")[-1].split(".")[0]
                os.system(
                    f"unzip {os.path.join(dataset_dir,file_name)}.zip -d {dataset_dir}"
                )
                # remove the tar.gz file
                os.system(f"rm {os.path.join(dataset_dir,file_name)}.zip")
                # rename the extracted folder from Animals_with_Attributes2 to the dataset name
                os.rename(
                    os.path.join(dataset_dir, "Animals_with_Attributes2"),
                    self._root_dir,
                )
            else:
                raise ValueError(f"Dataset is not a tar.gz file.")

            os.rename(
                os.path.join(self._root_dir, "classes.txt"),
                os.path.join(self._root_dir, "class_names.txt"),
            )
            os.rename(
                os.path.join(self._root_dir, "predicates.txt"),
                os.path.join(self._root_dir, "predicate_names.txt"),
            )

            # substract 1 from the class_id to make it 0-indexed
            predicate_names = []
            with open(os.path.join(self._root_dir, "predicate_names.txt"), "r") as f:
                predicate_names = [
                    f"{int(line.split()[0]) - 1} {line.split()[1]}\n" for line in f
                ]
            with open(os.path.join(self._root_dir, "predicate_names.txt"), "w") as f:
                f.writelines(predicate_names)

            class_names = {}
            with open(os.path.join(self._root_dir, "class_names.txt"), "r") as f:
                # key: class_name (str) value: class_id (int)
                # subtract 1 from the class_id to make it 0-indexed
                class_names = {line.split()[1]: int(line.split()[0]) - 1 for line in f}
            with open(os.path.join(self._root_dir, "class_names.txt"), "w") as f:
                # key: class_id (int) value: class_name (str)
                # save the adjusted class_id
                for class_name, class_id in class_names.items():
                    f.write(f"{class_id} {class_name}\n")

            # go through all subfolder in JPEGImages and save the sample names
            image_id = 0
            sample_labels = {}
            sample_names = {}
            for class_name in os.listdir(os.path.join(self._root_dir, "JPEGImages")):
                for sample_name in os.listdir(
                    os.path.join(self._root_dir, "JPEGImages", class_name)
                ):
                    relative_path = os.path.join(class_name, sample_name)
                    sample_names[image_id] = relative_path
                    sample_labels[image_id] = class_names[class_name]
                    image_id += 1

            with open(os.path.join(self._root_dir, "sample_names.txt"), "w") as f:
                # key: sample_id (int) value: sample_name (str)
                for sample_id, sample_name in sample_names.items():
                    f.write(f"{sample_id} {sample_name}\n")

            with open(os.path.join(self._root_dir, "sample_labels.txt"), "w") as f:
                # key: sample_id (int) value: class_id (int)
                for sample_id, class_id in sample_labels.items():
                    f.write(f"{sample_id} {class_id}\n")

            # rename image folder to samples
            os.rename(
                os.path.join(self._root_dir, "JPEGImages"),
                os.path.join(self._root_dir, "samples"),
            )

    def _load_dataset_info(self) -> None:
        super()._load_dataset_info()

        class_predicates = {}
        predicate_names = {}
        # load the class predicates
        with open(
            os.path.join(self._root_dir, "predicate-matrix-binary.txt"), "r"
        ) as f:
            class_predicates = {
                i: list(map(int, line.split())) for i, line in enumerate(f)
            }

        # load the predicate names
        with open(os.path.join(self._root_dir, "predicate_names.txt"), "r") as f:
            predicate_names = {int(line.split()[0]): line.split()[1] for line in f}

        # count the number of visual predicates
        for id, val in visual_predicates_dict.items():
            if val:
                self._vis_pred_ids.append(id)
                self._num_vis_predicates += 1

        # load the class visual predicates
        for class_id, predicates in class_predicates.items():
            self._class_vis_predicates[class_id] = [
                predicates[id] for id in self._vis_pred_ids
            ]

        # load the visual predicate names
        for i, id in enumerate(self._vis_pred_ids):
            self._vis_predicate_names[i] = predicate_names[id]

        # load train and test classes
        train_names = []
        with open(os.path.join(self._root_dir, "trainclasses.txt"), "r") as f:
            train_names = [line.strip() for line in f]

        test_names = []
        with open(os.path.join(self._root_dir, "testclasses.txt"), "r") as f:
            test_names = [line.strip() for line in f]

        for class_id, class_name in self._class_names.items():
            if class_name in train_names:
                self._train_classes[class_id] = class_name
            elif class_name in test_names:
                self._test_classes[class_id] = class_name

        # load the predicate weights
        self._predicate_weights = [0.0] * self._num_vis_predicates
        total_predicates = 0
        # sum the predicate values for each class
        for class_id, predicates in self._class_vis_predicates.items():
            for i, val in enumerate(predicates):
                self._predicate_weights[i] += val
                total_predicates += val

        for i, val in enumerate(self._predicate_weights):
            self._predicate_weights[i] = val / total_predicates

    def split_dataset(
        self,
        k: int = 2,
        seed: int = 42,
        shuffle: bool = True,
        stratified: bool = True,
        train_size: float = 0.7,
    ) -> None:
        # self.decompress_samples_folder()
        super().split_dataset(k, seed, shuffle, stratified, train_size)
        # self.compress_samples_folder()

    def predicate_table(self) -> pd.DataFrame:
        # make a table with rows as predicates and columns as classes
        rows = [val for val in self._vis_predicate_names.values()]
        cols = [val for val in self._class_names.values()]

        table = pd.DataFrame(index=rows, columns=cols)

        for class_id, predicates in self._class_vis_predicates.items():
            table.loc[:, self._class_names[class_id]] = predicates

        return table

    def fold_info(self, k: int, dir_name) -> dict:
        info = super().fold_info(k, dir_name)

        return self._add_specific_info(info)

    def test_info(self) -> dict:
        info = super().test_info()

        return self._add_specific_info(info)

    def _add_specific_info(self, info: dict) -> dict:
        # add visual predicates as labels
        if self._multi_label:
            info["class_ids"] = [class_ids for class_ids in info["labels"]]
            info["labels"] = [
                self._class_vis_predicates[class_ids[0]] for class_ids in info["labels"]
            ]

        return info
