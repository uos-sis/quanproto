import albumentations as A

pipnet_augmentation_ranges = {
    "small": {
        "hue": (-0.01, 0.01),
        "brightness": (0.90, 1.10),
        "saturation": (0.90, 1.10),
        "contrast": (0.90, 1.10),
        "posterize": (6, 8),
        "blur": (3, 3),
    },
    "medium": {
        "hue": (-0.025, 0.025),
        "brightness": (0.85, 1.15),
        "saturation": (0.85, 1.15),
        "contrast": (0.85, 1.15),
        "posterize": (4, 8),
        "blur": (3, 3),
    },
    "large": {  # best augmentation with geometric medium
        "hue": (-0.05, 0.05),
        "brightness": (0.75, 1.25),
        "saturation": (0.75, 1.25),
        "contrast": (0.75, 1.25),
        "posterize": (2, 8),
        "blur": (3, 3),
    },
}


def get_pipnet_augmentation_pipeline(range_id):
    return [
        A.ColorJitter(
            brightness=pipnet_augmentation_ranges[range_id]["brightness"],
            contrast=pipnet_augmentation_ranges[range_id]["contrast"],
            saturation=pipnet_augmentation_ranges[range_id]["saturation"],
            hue=pipnet_augmentation_ranges[range_id]["hue"],
            p=1.0,
        ),  # uses uniform distribution so 1.0
        A.Posterize(pipnet_augmentation_ranges[range_id]["posterize"], always_apply=True),
        A.Blur(blur_limit=pipnet_augmentation_ranges[range_id]["blur"], p=0.5),
    ]
