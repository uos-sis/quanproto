"""
This file contains the push method

The implementaion is based on the original ProtoPNet repository.
Reference: https://github.com/cfchen-duke/ProtoPNet
"""

import torch
import numpy as np
from tqdm import tqdm


def push_prototypes(
    model,  # pytorch network with prototype_vectors
    dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
) -> None:

    # Extract shape and number of prototypes from the network
    prototype_shape = model.prototype_shape
    n_prototypes = model.num_prototypes()

    # Initialize arrays to track the minimum distance and corresponding feature map patches for each prototype
    global_min_fmap_patches = np.zeros(
        [n_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3]]
    )

    data_iter = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Push Prototypes",
        mininterval=2.0,
        ncols=0,
    )

    push_log = {}
    # Iterate over the dataloader to update prototypes
    for i, (search_batch_input, search_y) in data_iter:
        push_batch_log = update_prototypes_on_batch(
            search_batch_input,
            model,
            global_min_fmap_patches,
            search_y=search_y,
            num_classes=model.num_labels,
        )
        # current img count
        img_count = i * dataloader.batch_size
        for proto_idx, log in push_batch_log.items():
            push_log[proto_idx] = {
                "img_id": int(dataloader.dataset.img_ids[log["img_index"] + img_count]),
                "patch_index": log["patch_index"],
                "min_dist": float(log["min_dist"]),
            }
    # sort the push log by key
    push_log = dict(sorted(push_log.items()))

    # Update the prototype vectors in the network
    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    model.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

    return push_log


def update_prototypes_on_batch(
    search_batch_input,
    model,
    global_min_fmap_patches,  # this will be updated
    search_y,  # required if class_specific == True
    num_classes,  # required if class_specific == True
    prototype_layer_stride=1,
):

    # Forward pass of the batch through the network
    with torch.no_grad():
        search_batch = search_batch_input.cuda()
        protoL_input_torch, proto_dist_torch = model.push_forward(search_batch)
        del search_batch  # Free up memory

    # Convert Torch tensors to NumPy arrays for processing
    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())
    del protoL_input_torch, proto_dist_torch  # Free up memory

    # create a dictionary mapping classes to image indices
    class_to_img_index_dict = {key: [] for key in range(num_classes)}
    for img_index, img_y in enumerate(search_y):
        if model.multi_label:
            img_labels = img_y.cpu().numpy()
            for i, label in enumerate(img_labels):
                if label == 1:
                    class_to_img_index_dict[i].append(img_index)
        else:
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    # Extract the shape and size information of prototypes
    prototype_shape = model.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h, proto_w = prototype_shape[2], prototype_shape[3]

    push_batch_log = {}
    # Iterate over each prototype to update its information
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    for j in range(n_prototypes):
        # Obtain target class of the prototype and continue only if images of that class exist in the batch
        target_class = torch.argmax(model.prototype_class_identity[j]).item()
        if len(class_to_img_index_dict[target_class]) == 0:
            continue
        proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j, :, :]

        # Find the minimum distance of the prototype to the current batch's data
        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = list(
                np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape)
            )

            # Adjust indices
            batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][
                batch_argmin_proto_dist_j[0]
            ]

            # Extract the feature map patch corresponding to the prototype
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w
            batch_min_fmap_patch_j = protoL_input_[
                img_index_in_batch,
                :,
                fmap_height_start_index:fmap_height_end_index,
                fmap_width_start_index:fmap_width_end_index,
            ]

            # Update global minimum distance and the corresponding feature map patch
            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            push_batch_log[j] = {
                "img_index": img_index_in_batch,
                # xmin, ymin, xmax, ymax
                "patch_index": (
                    int(fmap_width_start_index),
                    int(fmap_height_start_index),
                    int(fmap_width_end_index),
                    int(fmap_height_end_index),
                ),
                "min_dist": batch_min_proto_dist_j,
            }

    # Cleanup and finalization code
    del (class_to_img_index_dict,)

    return push_batch_log
