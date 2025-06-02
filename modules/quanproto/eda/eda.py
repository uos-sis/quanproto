import os
import math
import pandas as pd
import numpy as np
import skimage as ski
import multiprocessing
import matplotlib.pyplot as plt
import shutil
from skimage import io

figure_height = 5
figure_width = 8


def save_statistics(statistics: pd.DataFrame, folder: os.path, file_prefix: str) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)

    markdown_table = statistics.to_markdown()
    latex_table = statistics.to_latex()

    # delete if already exists
    if os.path.exists(os.path.join(folder, file_prefix + ".md")):
        os.remove(os.path.join(folder, file_prefix + ".md"))
    if os.path.exists(os.path.join(folder, file_prefix + ".tex")):
        os.remove(os.path.join(folder, file_prefix + ".tex"))

    # save to file
    with open(os.path.join(folder, file_prefix + ".md"), "w") as file:
        file.write(markdown_table)
    with open(os.path.join(folder, file_prefix + ".tex"), "w") as file:
        file.write(latex_table)


def save_class_histogram(
    counts: np.array,
    vals: np.array,
    folder: os.path,
    file_prefix: str,
    x_label="Label",
    y_label="Sample Count",
    bars: bool = False,
    save_md: bool = True,
) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    if bars:
        ax.bar(vals, counts)
    else:
        ax.plot(vals, counts)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Distribution")

    # delete if already exists
    if os.path.exists(os.path.join(folder, file_prefix + ".pdf")):
        os.remove(os.path.join(folder, file_prefix + ".pdf"))
    if os.path.exists(os.path.join(folder, file_prefix + ".png")):
        os.remove(os.path.join(folder, file_prefix + ".png"))

    # save to file
    plt.savefig(os.path.join(folder, file_prefix + ".pdf"))
    plt.savefig(os.path.join(folder, file_prefix + ".png"))
    plt.close()

    if save_md:
        # save as markdown table
        # round all counts to integers
        counts = np.round(counts, 0).astype(int)
        table = pd.DataFrame(counts, index=vals, columns=[y_label]).to_markdown()
        with open(os.path.join(folder, file_prefix + ".md"), "w") as file:
            file.write(table)


def save_color_histogram(
    counts: np.array, vals: np.array, folder: os.path, file_prefix: str
) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)

    if counts.shape[0] != 4:
        raise ValueError("The counts array must have 4 channels (RGB and Exposure)")
    if vals.shape[0] != counts.shape[1]:
        raise ValueError("The counts and vals arrays must have the same length")

    # plot histograms for each color channel
    _, ax = plt.subplots(1, 4, figsize=(figure_width * 4, figure_height))

    ax[0].plot(vals, counts[0], color="red")
    ax[0].set_title("Red Histogram")
    ax[0].set_xlabel("Pixel Value")
    ax[0].set_ylabel("Frequency")

    ax[1].plot(vals, counts[1], color="green")
    ax[1].set_title("Green Histogram")
    ax[1].set_xlabel("Pixel Value")
    ax[1].set_ylabel("Frequency")

    ax[2].plot(vals, counts[2], color="blue")
    ax[2].set_title("Blue Histogram")
    ax[2].set_xlabel("Pixel Value")
    ax[2].set_ylabel("Frequency")

    ax[3].plot(vals, counts[3], color="black")
    ax[3].set_title("Exposure Histogram")
    ax[3].set_xlabel("Pixel Value")
    ax[3].set_ylabel("Frequency")

    # delete if already exists
    if os.path.exists(os.path.join(folder, file_prefix + ".pdf")):
        os.remove(os.path.join(folder, file_prefix + ".pdf"))
    if os.path.exists(os.path.join(folder, file_prefix + ".png")):
        os.remove(os.path.join(folder, file_prefix + ".png"))

    plt.savefig(os.path.join(folder, file_prefix + ".pdf"))
    plt.savefig(os.path.join(folder, file_prefix + ".png"))
    plt.close()

    # plot histograms in one figure
    _, ax = plt.subplots(figsize=(figure_width, figure_height))

    colors = ["red", "green", "blue", "black"]
    labels = ["Red", "Green", "Blue", "Exposure"]

    for i in range(4):
        ax.plot(vals, counts[i], color=colors[i], label=labels[i])

    ax.set_title("Color Histograms")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.legend()

    # delete if already exists
    if os.path.exists(os.path.join(folder, file_prefix + "_combined.pdf")):
        os.remove(os.path.join(folder, file_prefix + "_combined.pdf"))
    if os.path.exists(os.path.join(folder, file_prefix + "_combined.png")):
        os.remove(os.path.join(folder, file_prefix + "_combined.png"))

    plt.savefig(os.path.join(folder, file_prefix + "_combined.pdf"))
    plt.savefig(os.path.join(folder, file_prefix + "_combined.png"))
    plt.close()


def stratified_cross_validation_statistics(sample_class_dic: dict) -> pd.DataFrame:
    """
    Calculate the min, mean, and max number of samples per class in a dataset for stratified cross validation splits
    with 2-5 folds.
    This is done for the following train-test splits: 50-50, 60-40, 70-30, 80-20, 90-10.
    Returns:
        pandas DataFrame with the statistics
    """
    # get the dictionary values as numpy array
    class_labels = np.array(list(sample_class_dic.values()))

    # use unique to get the number of samples per class
    _, class_counts = np.unique(class_labels, return_counts=True)

    train_test_splits = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1), (1.0, 0.0)]
    fold_splits = [2, 3, 4, 5]

    # Create a pandas DataFrame to store the statistics
    columns = []
    for i in fold_splits:
        columns.extend([f"train_{i}_folds", f"val_{i}_folds"])
    columns = ["split", "test", "train"] + columns

    index = []
    for _ in range(0, len(train_test_splits)):
        index.extend(["min", "mean", "max"])

    statistic = pd.DataFrame(columns=columns, index=index)

    # fill out the split column
    for i in range(0, len(train_test_splits)):
        statistic.iloc[i * 3, 0] = (
            f"{int(train_test_splits[i][0]*100)}-{int(train_test_splits[i][1]*100)}"
        )
        statistic.iloc[i * 3 + 1, 0] = (
            f"{int(train_test_splits[i][0]*100)}-{int(train_test_splits[i][1]*100)}"
        )
        statistic.iloc[i * 3 + 2, 0] = (
            f"{int(train_test_splits[i][0]*100)}-{int(train_test_splits[i][1]*100)}"
        )

    max_count = float(max(class_counts))
    min_count = float(min(class_counts))
    mean_count = float(np.mean(class_counts))

    ndigits = 0
    for i in range(0, len(train_test_splits)):
        split = train_test_splits[i]

        # fill test statistics
        statistic.iloc[i * 3, 1] = math.ceil(min_count * split[1])
        statistic.iloc[i * 3 + 1, 1] = math.ceil(mean_count * split[1])
        statistic.iloc[i * 3 + 2, 1] = math.ceil(max_count * split[1])

        train_max = float(math.floor(max_count * split[0]))
        train_min = float(math.floor(min_count * split[0]))
        train_mean = float(math.floor(mean_count * split[0]))

        # fill train statistics
        statistic.iloc[i * 3, 2] = int(train_min)
        statistic.iloc[i * 3 + 1, 2] = int(train_mean)
        statistic.iloc[i * 3 + 2, 2] = int(train_max)

        # fill fold statistics
        for fold in fold_splits:
            # train
            statistic.iloc[i * 3, 3 + (fold - 2) * 2] = math.floor(
                train_min / float(fold) * float(fold - 1)
            )
            statistic.iloc[i * 3 + 1, 3 + (fold - 2) * 2] = math.floor(
                train_mean / float(fold) * float(fold - 1)
            )
            statistic.iloc[i * 3 + 2, 3 + (fold - 2) * 2] = math.floor(
                train_max / float(fold) * float(fold - 1)
            )

            # val
            statistic.iloc[i * 3, 4 + (fold - 2) * 2] = math.ceil(train_min / float(fold))
            statistic.iloc[i * 3 + 1, 4 + (fold - 2) * 2] = math.ceil(train_mean / float(fold))
            statistic.iloc[i * 3 + 2, 4 + (fold - 2) * 2] = math.ceil(train_max / float(fold))

    return statistic


def mask_statistics(num_masks_x_labels: np.array) -> pd.DataFrame:
    # Get number of masks and associated class labels seperately
    sample_num_masks = num_masks_x_labels[:, 0]
    sample_labels = num_masks_x_labels[:, 1]

    # Get all labels to calculate "per class"-statistics
    all_labels = np.unique(sample_labels)

    # get the number of masks per class
    num_masks_per_class = np.array(
        [sample_num_masks[sample_labels == label].sum() for label in all_labels]
    )

    statistic = pd.DataFrame(columns=["statistics"])

    statistic.loc["num samples", "statistics"] = len(sample_num_masks)
    statistic.loc["num classes", "statistics"] = np.unique(sample_labels).size
    statistic.loc["min masks/image", "statistics"] = np.min(sample_num_masks)
    statistic.loc["mean masks/image", "statistics"] = round(np.mean(sample_num_masks))
    statistic.loc["max masks/image", "statistics"] = np.max(sample_num_masks)
    statistic.loc["std masks/image", "statistics"] = round(np.std(sample_num_masks))

    statistic.loc["min masks/class", "statistics"] = np.min(num_masks_per_class)
    statistic.loc["mean masks/class", "statistics"] = round(np.mean(num_masks_per_class))
    statistic.loc["max masks/class", "statistics"] = np.max(num_masks_per_class)
    statistic.loc["std masks/class", "statistics"] = round(np.std(num_masks_per_class))

    return statistic


def class_statistics(class_labels: np.array) -> pd.DataFrame:
    # use unique to get the number of samples per class
    _, class_counts = np.unique(class_labels, return_counts=True)

    statistic = pd.DataFrame(columns=["statistics"])

    statistic.loc["num samples", "statistics"] = len(class_labels)
    statistic.loc["num classes", "statistics"] = len(class_counts)
    statistic.loc["min sample/class", "statistics"] = min(class_counts)
    statistic.loc["mean sample/class", "statistics"] = round(np.mean(class_counts))
    statistic.loc["max sample/class", "statistics"] = max(class_counts)
    statistic.loc["std sample/class", "statistics"] = round(np.std(class_counts), ndigits=2)

    return statistic


def class_histogram(class_labels: np.array) -> tuple[np.array, np.array]:
    """
    Calculate the histogram of the classes in a dataset.
    """
    # use unique to get the number of samples per class
    class_labels, class_counts = np.unique(class_labels, return_counts=True)

    return class_counts, class_labels


def mask_histogram(num_masks_x_labels: np.array) -> tuple[np.array, np.array]:
    # Get class labels associated to masks
    sample_labels = num_masks_x_labels[:, 1]

    # Get all existing labels
    all_labels = np.unique(sample_labels)

    # To detect classes with generally high/low number of masks
    num_masks_per_sample_per_label = np.array(
        [num_masks_x_labels[sample_labels == label, 0].mean() for label in all_labels]
    )

    return num_masks_per_sample_per_label, all_labels


def image_color_distribution(image_dir: os.path) -> np.array:
    try:
        image = ski.io.imread(image_dir)
    except Exception as e:
        print(e)
        print(f"Error reading image: {image_dir}")
        os.makedirs(os.path.join(os.path.dirname(image_dir), "failed"), exist_ok=True)
        shutil.move(image_dir, os.path.join(os.path.dirname(image_dir), "failed"))
        return np.zeros((3, 256))

    # check if the image is grayscale
    if len(image.shape) == 2:
        # convert to 3 channels
        image = ski.color.gray2rgb(image)

    if image.shape[2] == 4:
        # RGBA to RGB
        image = ski.color.rgba2rgb(image)

    if image.shape[2] != 3:
        return np.zeros((3, 256))

    # remsize the image
    image = ski.transform.resize(image, (224, 224), anti_aliasing=True)

    # normalize the image
    image = ski.img_as_float(image)
    # use numpy to calculate the histogram
    red_count, _ = np.histogram(image[:, :, 0], bins=256, range=(0, 1))
    green_count, _ = np.histogram(image[:, :, 1], bins=256, range=(0, 1))
    blue_count, _ = np.histogram(image[:, :, 2], bins=256, range=(0, 1))

    counts = np.array([red_count, green_count, blue_count])

    return counts


def color_histogram(image_dir: os.path) -> tuple[np.array, np.array]:
    """
    Calculate the color distribution of the images in a directory.

    """
    # get the image file names
    all_files = []

    for root, directories, files in os.walk(image_dir):
        for file in files:
            # check if jpg or png
            if file.endswith(".jpg") or file.endswith(".png"):
                file_path = os.path.join(root, file)
                all_files.append(file_path)

    if len(all_files) == 0:
        raise ValueError("No images found in the directory")

    # Create a Pool of processes
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        histograms = pool.map(image_color_distribution, all_files)

    # histograms = []
    # for file in all_files:
    #     histograms.append(image_color_distribution(file))

    # Specify the data range
    data_min = 0
    data_max = 1
    # Calculate the bin edges
    bin_edges = np.linspace(data_min, data_max, 256 + 1)
    # Calculate the bin centers
    vals = (bin_edges[:-1] + bin_edges[1:]) / 2

    total_histogram = np.sum(histograms, axis=0)

    # calculate the grayscale channel using the luminance coefficients
    grayscale_channel = (
        0.2989 * total_histogram[0] + 0.5870 * total_histogram[1] + 0.1140 * total_histogram[2]
    )

    # reshape to (1, bins)
    grayscale_channel = grayscale_channel.reshape(1, -1)

    # append the grayscale channel to the histograms
    total_histogram = np.append(total_histogram, grayscale_channel, axis=0)

    return total_histogram, vals


def color_statistics(counts: np.array, vals: np.array) -> pd.DataFrame:
    """
    Calculate the mean and variance of the color channels in a directory.
    """
    # calculate the mean and variance
    val_his = counts * vals
    total_count = np.sum(counts, axis=1, keepdims=True)
    mean = np.sum(val_his, axis=1, keepdims=True) / total_count
    variance = np.sum((vals - mean) ** 2 * counts, axis=1, keepdims=True) / total_count

    # create a pandas DataFrame
    columns = ["red", "green", "blue", "exposure"]
    statistics = pd.DataFrame(columns=columns)
    statistics.loc["mean", :] = mean.flatten()
    statistics.loc["variance", :] = variance.flatten()

    return statistics


def segmentation_folder_statistics(
    num_masks_x_labels: np.array,
    image_dir: os.path,
    log_dir: os.path,
    prefix: str | None = None,
    bars: bool = False,
) -> None:
    # Masks Statistics
    overview = mask_statistics(num_masks_x_labels)
    if prefix is None:
        save_statistics(overview, log_dir, "mask_statistics")
    else:
        save_statistics(overview, log_dir, f"{prefix}_mask_statistics")

    # Masks Distribution
    counts, labels = mask_histogram(num_masks_x_labels)
    if prefix is None:
        save_class_histogram(
            counts, labels, log_dir, "mask_class_distribution", y_label="Mask Mean", bars=bars
        )
    else:
        save_class_histogram(
            counts,
            labels,
            log_dir,
            f"{prefix}_mask_class_distribution",
            y_label="Mask Mean",
            bars=bars,
        )


def segmentation_size_statistics(
    mask_sizes: list,
    folder: os.path,
    prefix: str | None = None,
    bars: bool = False,
    save_md: bool = True,
) -> None:
    """computes the size statistics of the segmentation masks"""
    # get the number of masks per image
    num_masks = np.array([len(masks) for masks in mask_sizes])

    max_masks = np.max(num_masks)

    mask_index_counts = np.zeros(max_masks)
    mask_size_sums = np.zeros(max_masks)

    for i in range(num_masks.size):
        mask_index_counts[: num_masks[i]] += 1
        mask_size_sums[: num_masks[i]] += mask_sizes[i][: num_masks[i]]

    assert np.all(mask_index_counts > 0)

    mask_size_means = mask_size_sums / mask_index_counts

    if save_md:
        # save as markdown table
        table = pd.DataFrame(
            mask_size_means, index=np.arange(1, max_masks + 1), columns=["Mean Mask Size"]
        ).to_markdown()
        with open(os.path.join(folder, prefix + "_size_distribution" + ".md"), "w") as file:
            file.write(table)

    threshold = 0.01
    # search for the first index where the mean mask size is below the threshold
    thres_idx = np.where(mask_size_means < threshold)[0]
    if thres_idx.size == 0:
        thres_idx = max_masks
    else:
        thres_idx = thres_idx[0]
    mask_size_means = mask_size_means[:thres_idx]
    counts = np.arange(1, thres_idx + 1)

    if not os.path.exists(folder):
        os.makedirs(folder)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    if bars:
        # ax.bar(np.arange(1, max_masks + 1), mask_size_means)
        ax.bar(counts, mask_size_means)
    else:
        ax.plot(counts, mask_size_means)

    plt.xlabel("Mask Index")
    plt.ylabel("Mean Mask Size")
    plt.title("Distribution")

    # delete if already exists
    if os.path.exists(os.path.join(folder, prefix + "_size_distribution" + ".pdf")):
        os.remove(os.path.join(folder, prefix + "_size_distribution" + ".pdf"))
    if os.path.exists(os.path.join(folder, prefix + "_size_distribution" + ".png")):
        os.remove(os.path.join(folder, prefix + "_size_distribution" + ".png"))

    # save to file
    plt.savefig(os.path.join(folder, prefix + "_size_distribution" + ".pdf"))
    plt.savefig(os.path.join(folder, prefix + "_size_distribution" + ".png"))
    plt.close()


def process_masks(object_mask_path, segment_mask_paths, num_masks, max_masks):

    local_mask_counts = np.zeros(max_masks)
    local_mask_sums = np.zeros(max_masks)

    # Get the object mask
    object_mask = io.imread(object_mask_path)
    if len(object_mask.shape) == 3:
        object_mask = np.sum(object_mask, axis=2) > 0

    if len(object_mask.shape) == 3:
        object_mask = np.sum(object_mask, axis=2) > 0

    for j in range(num_masks):
        segment_mask = io.imread(segment_mask_paths[j])
        segment_mask = np.sum(segment_mask, axis=2) > 0

        assert (
            object_mask.shape == segment_mask.shape
        ), f"mask shapes do not match {object_mask.shape} != {segment_mask.shape}"

        # Calculate the overlap
        intersection = np.sum(np.logical_and(object_mask, segment_mask))
        union = np.sum(segment_mask > 0)
        if union == 0:
            overlap = 0
            print(f"mask is empty: {segment_mask_paths[j]}")
        else:
            overlap = intersection / union

        local_mask_counts[j] += 1
        local_mask_sums[j] += overlap

    return local_mask_counts, local_mask_sums


def segmentation_object_overlap_statistics(
    dataset,
    method: str,
    folder: os.path,
    prefix: str | None = None,
    bars: bool = False,
    save_md: bool = True,
) -> None:

    object_masks = dataset.sample_info()["masks"]["original"]
    segment_masks = dataset.sample_info()["masks"][method]

    # get the number of masks per image
    num_masks = np.array([len(masks) for masks in segment_masks["paths"]])

    max_masks = np.max(num_masks)

    mask_index_counts = np.zeros(max_masks)
    mask_overlap_sums = np.zeros(max_masks)

    all_args = []
    for i in range(num_masks.size):
        segmentation_paths = [
            os.path.join(dataset._segmentation_dir, method, path)
            for path in segment_masks["paths"][i]
        ]
        all_args.append(
            (
                os.path.join(dataset._segmentation_dir, "original", object_masks["paths"][i][0]),
                segmentation_paths,
                num_masks[i],
                max_masks,
            )
        )

    with Pool() as pool:
        results = pool.starmap(process_masks, all_args)

    # Combine results
    for counts, sums in results:
        mask_index_counts += counts
        mask_overlap_sums += sums

    mask_overlap_means = mask_overlap_sums / mask_index_counts

    counts = np.arange(1, max_masks + 1)

    if not os.path.exists(folder):
        os.makedirs(folder)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    if bars:
        # ax.bar(np.arange(1, max_masks + 1), mask_size_means)
        ax.bar(counts, mask_overlap_means)
    else:
        ax.plot(counts, mask_overlap_means)

    plt.xlabel("Mask Index")
    plt.ylabel("Mean Object Overlap")
    plt.title("Distribution")

    # delete if already exists
    if os.path.exists(os.path.join(folder, prefix + "_overlap_distribution" + ".pdf")):
        os.remove(os.path.join(folder, prefix + "_overlap_distribution" + ".pdf"))
    if os.path.exists(os.path.join(folder, prefix + "_overlap_distribution" + ".png")):
        os.remove(os.path.join(folder, prefix + "_overlap_distribution" + ".png"))

    # save to file
    plt.savefig(os.path.join(folder, prefix + "_overlap_distribution" + ".pdf"))
    plt.savefig(os.path.join(folder, prefix + "_overlap_distribution" + ".png"))
    plt.close()

    if save_md:
        # save as markdown table
        table = pd.DataFrame(
            mask_overlap_means, index=np.arange(1, max_masks + 1), columns=["Mean Mask Size"]
        ).to_markdown()
        with open(os.path.join(folder, prefix + "_overlap_distribution" + ".md"), "w") as file:
            file.write(table)


def image_folder_statistics(
    sample_class_labels: np.array,
    image_dir: os.path,
    log_dir: os.path,
    prefix: str | None = None,
    bars: bool = False,
) -> None:
    """
    Calculate the statistics and histograms of the images in a directory and save them to a log directory.
    """
    # Class Statistics
    overview = class_statistics(sample_class_labels)
    if prefix is None:
        save_statistics(overview, log_dir, "class_statistics")
    else:
        save_statistics(overview, log_dir, f"{prefix}_class_statistics")

    # Class Distribution
    counts, labels = class_histogram(sample_class_labels)
    if prefix is None:
        save_class_histogram(counts, labels, log_dir, "class_distribution", bars=bars)
    else:
        save_class_histogram(counts, labels, log_dir, f"{prefix}_class_distribution", bars=bars)

    # Image Histograms
    counts, vals = color_histogram(image_dir)
    norm_counts = counts / np.sum(counts, axis=1, keepdims=True)
    if prefix is None:
        save_color_histogram(norm_counts, vals, log_dir, "color_histograms")
    else:
        save_color_histogram(norm_counts, vals, log_dir, f"{prefix}_color_histograms")

    # Image Statistics
    statistics = color_statistics(counts, vals)
    if prefix is None:
        save_statistics(statistics, log_dir, "image_statistics")
    else:
        save_statistics(statistics, log_dir, f"{prefix}_image_statistics")
