def get_dataset(dataset_dir: str, dataset: str, **kwargs):
    if dataset == "cub200":
        from . import cub200

        return cub200.CUB200(dataset_dir, **kwargs)
    elif dataset == "cub200_mini":
        from . import cub200

        return cub200.CUB200(dataset_dir, **kwargs)
    elif dataset == "cars196":
        from . import cars196

        return cars196.Cars196(dataset_dir, **kwargs)

    elif dataset == "dogs":
        from . import dogs

        return dogs.DOGS(dataset_dir, **kwargs)
    elif dataset == "dogs_mini":
        from . import dogs

        return dogs.DOGS(dataset_dir, **kwargs)
    elif dataset == "awa2":
        from . import awa2

        return awa2.AwA2(dataset_dir, **kwargs)
    elif dataset == "nico":
        from . import nico

        return nico.Nico(dataset_dir, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


dogs_params = {
    "num_classes": 120,
    "multi_label": False,
}

dogs_mini_params = {
    "num_classes": 10,
    "multi_label": False,
}

cub200_params = {
    "num_classes": 200,
    "multi_label": False,
}

cub200_mini_params = {
    "num_classes": 10,
    "multi_label": False,
}

cars196_params = {
    "num_classes": 196,
    "multi_label": False,
}

awa2_params = {
    "num_classes": 49,
    "multi_label": True,
}

nico_params = {
    "num_classes": 10,  # INFO: or 10 if you use only the animals classes
    "multi_label": False,
}
