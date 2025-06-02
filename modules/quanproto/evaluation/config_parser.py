import quanproto.dataloader.complexity as complexity_dataloader
import quanproto.dataloader.continuity as continuity_dataloader
import quanproto.dataloader.multi_label as multi_label_dataloader
from quanproto.dataloader.params import BATCH_SIZE, NUM_DATALOADER_WORKERS, PIN_MEMORY
from quanproto.dataloader.single_augmentation import (
    test_dataloader,
    validation_dataloader,
)
from quanproto.techniques import (
    compactness,
    complexity,
    continuity,
    contrastivity,
    general,
    output_completeness,
    threshold,
    topk_prototype,
)

# region evaluation_techniques_fn snippet
evaluation_techniques_fn = {
    "contrastivity": contrastivity.evaluate_contrastivity,
    "general": general.evaluate_general,
    "continuity": continuity.evaluate_continuity,
    "compactness": compactness.evaluate_compactness,
    "output_completeness": output_completeness.evaluate_output_completeness,
    "complexity": complexity.evaluate_complexity,
    "topk_prototype_images": topk_prototype.topk_prototype_images,
    "threshold": threshold.evaluate_threshold,
}
# endregion evaluation_techniques_fn snippet


def get_dataloader_fn_dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_DATALOADER_WORKERS,
    pin_memory=PIN_MEMORY,
    crop=False,
    multi_label=False,
    validation_set=False,
):
    dataloder_to_use = validation_dataloader if validation_set else test_dataloader

    multi_label_dataloader_to_use = (
        multi_label_dataloader.validation_dataloader
        if validation_set
        else multi_label_dataloader.test_dataloader
    )

    continuity_dataloader_to_use = (
        continuity_dataloader.validation_dataloader
        if validation_set
        else continuity_dataloader.test_dataloader
    )

    complexity_dataloader_to_use = (
        complexity_dataloader.validation_dataloader
        if validation_set
        else complexity_dataloader.test_dataloader
    )

    dataloader_fn_dict = {
        "threshold": {
            "fn": dataloder_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
            },
        },
        "general": {
            "fn": dataloder_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
            },
        },
        "compactness": {
            "fn": dataloder_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
            },
        },
        "contrastivity": {
            "fn": multi_label_dataloader_to_use if multi_label else dataloder_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
            },
        },
        "continuity": {
            "fn": continuity_dataloader_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
            },
        },
        "output_completeness": {
            "fn": dataloder_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
            },
        },
        "complexity": {
            "fn": complexity_dataloader_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
            },
        },
        "topk_prototype_images": {
            "fn": dataloder_to_use,
            "args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "crop": crop,
            },
        },
    }

    return dataloader_fn_dict
