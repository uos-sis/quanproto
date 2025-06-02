"""
This file contains the augmentation pipelines for the different types of augmentations.
"""

import albumentations as A

import quanproto.models.pipnet.best_augmentation as pipnet_augmentation

geometric_augmentation_ranges = {
    "small": {
        "scale": (0.90, 1.10),
        "shear": (-10, 10),
        "rotation": (-10, 10),
        "translate": (-0.10, 0.10),
        "perspective": (0.0, 0.025),
    },
    "medium": {
        "scale": (0.85, 1.15),
        "shear": (-15, 15),
        "rotation": (-15, 15),
        "translate": (-0.15, 0.15),
        "perspective": (0.0, 0.05),
    },
    "large": {
        "scale": (0.75, 1.25),
        "shear": (-25, 25),
        "rotation": (-25, 25),
        "translate": (-0.25, 0.25),
        "perspective": (0.0, 0.1),
    },
}


def get_geometric_augmentation_pipeline(range_id: str = "medium"):
    """
    Returns the geometric augmentation pipeline.
    """
    return [
        A.HorizontalFlip(p=0.5),  # 50% makes a uniform distribution of the flip
        A.Perspective(
            scale=geometric_augmentation_ranges[range_id]["perspective"], p=1.0
        ),  # uses uniform distribution so 1.0
        A.Affine(
            scale=geometric_augmentation_ranges[range_id]["scale"],
            translate_percent=geometric_augmentation_ranges[range_id]["translate"],
            rotate=geometric_augmentation_ranges[range_id]["rotation"],
            shear=geometric_augmentation_ranges[range_id]["shear"],
            p=1.0,
            rotate_method="ellipse",
            mode=3,
        ),
    ]


photometric_augmentation_ranges = {
    "small": {
        "hue": (-0.01, 0.01),
        "brightness": (0.90, 1.10),
        "saturation": (0.90, 1.10),
        "contrast": (0.90, 1.10),
    },
    "medium": {
        "hue": (-0.025, 0.025),
        "brightness": (0.85, 1.15),
        "saturation": (0.85, 1.15),
        "contrast": (0.85, 1.15),
    },
    "large": {
        "hue": (-0.05, 0.05),
        "brightness": (0.75, 1.25),
        "saturation": (0.75, 1.25),
        "contrast": (0.75, 1.25),
    },
}


def get_photometric_augmentation_pipeline(range_id: str = "medium"):
    """
    Returns the photometric augmentation
    """
    return [
        A.ColorJitter(
            brightness=photometric_augmentation_ranges[range_id]["brightness"],
            contrast=photometric_augmentation_ranges[range_id]["contrast"],
            saturation=photometric_augmentation_ranges[range_id]["saturation"],
            hue=photometric_augmentation_ranges[range_id]["hue"],
            p=1.0,
        ),  # uses uniform distribution so 1.0
    ]


noise_augmentation_ranges = {
    "small": {
        "blur": (3, 3),
        "gaussian_noise": (0, 5),
        "jpeg_compression_quality": (90, 100),
    },
    "medium": {
        "blur": (5, 5),
        "gaussian_noise": (0, 10),
        "jpeg_compression_quality": (80, 100),
    },
    "large": {
        "blur": (7, 7),
        "gaussian_noise": (0, 15),
        "jpeg_compression_quality": (70, 100),
    },
}


def get_noise_augmentation_pipeline(range_id: str = "medium"):
    """
    Returns the noise augmentation pipeline.
    """
    return [
        A.GaussNoise(
            var_limit=noise_augmentation_ranges[range_id]["gaussian_noise"], p=1.0
        ),
        A.Blur(blur_limit=noise_augmentation_ranges[range_id]["blur"], p=1.0),
        A.ImageCompression(
            quality_lower=noise_augmentation_ranges[range_id][
                "jpeg_compression_quality"
            ][0],
            quality_upper=noise_augmentation_ranges[range_id][
                "jpeg_compression_quality"
            ][1],
            p=1.0,
        ),
    ]


def get_continuity_augmentation_pipeline(range_id: str = "medium"):
    """
    Returns the continuity augmentation pipeline.
    """
    return [
        A.ColorJitter(
            brightness=(1.125, 1.125),
            contrast=(1.125, 1.125),
            saturation=(1.125, 1.125),
            hue=(0.05, 0.05),
            p=1.0,
        ),
        A.GaussNoise(var_limit=(5, 5), p=1.0),
        A.Blur(blur_limit=(3, 3), p=1.0),
        A.ImageCompression(
            quality_lower=90,
            quality_upper=90,
            p=1.0,
        ),
    ]


augmentation_pipeline_dict = {
    "geometric": get_geometric_augmentation_pipeline,
    "photometric": get_photometric_augmentation_pipeline,
    "noise": get_noise_augmentation_pipeline,
    "continuity": get_continuity_augmentation_pipeline,
    "pipnet": pipnet_augmentation.get_pipnet_augmentation_pipeline,
}


def get_augmentation_pipeline(augmentation_key: str, range_id: str = "medium"):
    """
    Returns the augmentation pipeline based on the key.
    """
    if augmentation_key == "none":
        return [A.NoOp(p=1.0)]

    if augmentation_key == "crop":
        return [
            A.RandomCropNearBBox(max_part_shift=0.0, p=1.0),
        ]

    assert (
        augmentation_key in augmentation_pipeline_dict
    ), f"Unknown augmentation key: {augmentation_key}"

    # if the key was not from the above, then it must be one of the augmentation pipelines
    return augmentation_pipeline_dict[augmentation_key](range_id)
