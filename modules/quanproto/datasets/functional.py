import numpy as np
import sklearn.model_selection as sk
import shutil


def move_sample(sample_path, output_path):
    try:
        shutil.move(sample_path, output_path)
    except Exception as e:
        print(f"Error copying {sample_path} to {output_path}: {e}")


def get_random_bounding_box(bounding_boxes):
    """
    list of bounding boxes [[x_min, y_min, x_max, y_max], ...] (float)
    Returns a random bounding box [x_min, y_min, x_max, y_max] (float)
    """
    return bounding_boxes[np.random.randint(0, len(bounding_boxes))]


def combine_bounding_boxes(bounding_boxes):
    """
    list of bounding boxes [[x_min, y_min, x_max, y_max], ...] (float)
    Returns a list containing the combined bounding box [x_min, y_min, x_max, y_max] (float)
    """
    x_min = min([bb[0] for bb in bounding_boxes])
    y_min = min([bb[1] for bb in bounding_boxes])
    x_max = max([bb[2] for bb in bounding_boxes])
    y_max = max([bb[3] for bb in bounding_boxes])

    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def crop_bounding_box(shape, bounding_box) -> None:
    for box in bounding_box:
        if box[0] < 0.0:
            box[0] = 0.0
        if box[1] < 0.0:
            box[1] = 0.0
        if box[2] > shape[1]:
            box[2] = shape[1]
        if box[3] > shape[0]:
            box[3] = shape[0]


def copy_sample(sample_path, output_path):
    try:
        shutil.copy(sample_path, output_path)
    except Exception as e:
        print(f"Error copying {sample_path} to {output_path}: {e}")


def train_test_split(id_labels, train_size=0.7, seed=42, shuffle=True, stratified=True):
    """
    Split the dataset into train and test sets

    Args:
        id_labels: dictionary containing key:sample IDs and value:labels
        train_size: the proportion of the dataset to include in the train split
        seed: random seed for reproducibility
        stratified: whether to stratify the train/test split by label

    Returns:
        train_sample_ids: sample IDs for the training set as numpy array
        test_sample_ids: sample IDs for the test set as numpy array
    """
    # Get the sample IDs and labels
    sample_ids = np.array(list(id_labels.keys()))
    labels = np.array(list(id_labels.values()))

    # Split the dataset into train and test sets
    if stratified:
        train_sample_ids, test_sample_ids, _, _ = sk.train_test_split(
            sample_ids,
            labels,
            train_size=train_size,
            random_state=seed,
            shuffle=shuffle,
            stratify=labels,
        )
    else:
        train_sample_ids, test_sample_ids = sk.train_test_split(
            sample_ids, train_size=train_size, random_state=seed, shuffle=shuffle
        )

    return (train_sample_ids, test_sample_ids)


def k_fold_split(id_labels, n_splits=5, seed=42, shuffle=True, stratified=True):
    """
    Split the dataset into train and test sets

    Args:
        id_labels: dictionary containing key:sample IDs and value:labels
        n_splits: number of folds
        seed: random seed for reproducibility
        stratified: whether to stratify the train/test split by label

    Returns:
        list of tuples containing train and test indices for each fold as numpy arrays
    """
    # Get the sample IDs and labels
    sample_ids = np.array(list(id_labels.keys()))
    labels = np.array(list(id_labels.values()))

    # Split the dataset into train and test sets
    if stratified:
        kfold = sk.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    else:
        kfold = sk.KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    splits = []
    for train_indices, test_indices in kfold.split(sample_ids, labels):
        train_ids = sample_ids[train_indices]
        test_ids = sample_ids[test_indices]
        splits.append((train_ids, test_ids))

    return splits


def create_splits(
    sample_labels: dict,
    k: int = 3,
    seed: int = 42,
    shuffle: bool = True,
    stratified: bool = True,
    train_size: float = 0.7,
) -> tuple[list, np.array]:
    if k < 1:
        raise ValueError("Number of folds must be greater than 0")

    train_splits = []
    test_set = None

    # get the test set
    train_set, test_set = train_test_split(
        sample_labels,
        train_size=train_size,
        seed=seed,
        shuffle=shuffle,
        stratified=stratified,
    )
    train_splits = [(train_set, np.array([]))]

    if k > 1:
        # create a dictionary containing only the training samples
        subset_sample_labels = {
            sample_id: class_id
            for sample_id, class_id in sample_labels.items()
            if sample_id in train_set
        }

        train_splits = k_fold_split(
            subset_sample_labels,
            n_splits=k,
            seed=seed,
            shuffle=shuffle,
            stratified=stratified,
        )

    return train_splits, test_set
