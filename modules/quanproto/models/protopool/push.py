"""
This file contains the push method
The implementation is based on the original ProtoPool repository

Reference: https://github.com/gmum/ProtoPool

"""

import torch
import numpy as np


def push_prototypes(
    model,  # pytorch network with prototype_vectors
    dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
) -> None:
    global_min_fmap_patches = np.zeros(
        [
            model.num_prototypes(),
            model.prototype_shape[1],
            model.prototype_shape[2],
            model.prototype_shape[3],
        ]
    )

    for search_batch_input, search_y in dataloader:
        """
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        """
        update_prototypes_on_batch(
            search_batch_input=search_batch_input,
            model=model,
            global_min_fmap_patches=global_min_fmap_patches,
            search_y=search_y,
            prototype_layer_stride=1,
        )

    prototype_update = np.reshape(global_min_fmap_patches, tuple(model.prototype_shape))
    model.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())


def update_prototypes_on_batch(
    search_batch_input,
    model,
    global_min_fmap_patches,  # this will be updated
    search_y=None,
    prototype_layer_stride=1,
):
    model.eval()
    search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = model.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    prototype_shape = model.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]

    map_class_to_prototypes = model.get_map_class_to_prototypes()
    protype_to_img_index_dict = {key: [] for key in range(n_prototypes)}
    # img_y is the image's integer label

    for img_index, img_y in enumerate(search_y):
        if model.multi_label:
            img_labels = img_y.cpu().numpy()
            for class_idx, label in enumerate(img_labels):
                if label == 1:
                    [
                        protype_to_img_index_dict[prototype].append(img_index)
                        for prototype in map_class_to_prototypes[class_idx]
                    ]
        else:
            img_label = img_y.item()
            [
                protype_to_img_index_dict[prototype].append(img_index)
                for prototype in map_class_to_prototypes[img_label]
            ]

    global_min_proto_dist = np.full(n_prototypes, np.inf)
    for j in range(n_prototypes):
        if len(protype_to_img_index_dict[j]) == 0:
            continue
        proto_dist_j = proto_dist_[protype_to_img_index_dict[j]][:, j]

        batch_min_proto_dist_j = np.amin(proto_dist_j)

        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = list(
                np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape)
            )
            batch_argmin_proto_dist_j[0] = protype_to_img_index_dict[j][
                batch_argmin_proto_dist_j[0]
            ]

            # retrieve the corresponding feature map patch
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

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j
