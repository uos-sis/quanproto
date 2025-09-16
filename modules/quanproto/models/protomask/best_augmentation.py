import albumentations as A

protomask_augmentation_ranges = {
    "small": {
        "shear": (-10, 10),
        "rotation": (-10, 10),
        "perspective": (0.0, 0.025),
    },
    "medium": {
        "shear": (-15, 15),
        "rotation": (-15, 15),
        "perspective": (0.0, 0.05),
    },
    "large": {
        "shear": (-25, 25),
        "rotation": (-25, 25),
        "perspective": (0.0, 0.1),
    },
}


def get_protomask_augmentation_pipeline(range_id):
    return [
        A.HorizontalFlip(p=0.5),  # 50% makes a uniform distribution of the flip
        A.Perspective(
            scale=protomask_augmentation_ranges[range_id]["perspective"], p=1.0
        ),  # uses uniform distribution so 1.0
        A.Affine(
            rotate=protomask_augmentation_ranges[range_id]["rotation"],
            shear=protomask_augmentation_ranges[range_id]["shear"],
            p=1.0,
            rotate_method="ellipse",
            mode=3,
        ),
    ]
