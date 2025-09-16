"""
This file contains the pruning method.

The implementaion is based on the original ProtoPNet repository.
Reference: https://github.com/cfchen-duke/ProtoPNet
"""

import heapq
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm


class ImagePatchInfo:

    def __init__(self, label, distance):
        self.label = label
        self.negative_distance = -distance

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance


def find_k_nearest_patches_to_prototypes(
    dataloader,
    model,
    k=6,
):
    n_prototypes = model.num_prototypes()
    heaps = []
    # allocate an array of n_prototypes number of heaps
    for _ in range(n_prototypes):
        # a heap in python is just a maintained list
        heaps.append([])

    iter = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Finding k-nearest patches to prototypes",
        mininterval=2.0,
        ncols=0,
    )

    for i, (search_batch, search_y) in iter:

        with torch.no_grad():
            search_batch = search_batch.cuda()
            _, proto_dist_torch = model.push_forward(search_batch)

        proto_dist_ = np.copy(
            proto_dist_torch.detach().cpu().numpy()
        )  # batch, mask, proto_num

        for img_idx, distance_map in enumerate(proto_dist_):
            # change shape from (mask, proto_num) to (proto_num, mask)
            distance_map = distance_map.T

            for j in range(n_prototypes):
                # find the closest patches in this batch to prototype j
                closest_patch_distance_to_prototype_j = np.amin(distance_map[j])

                closest_patch = ImagePatchInfo(
                    label=search_y[img_idx],
                    distance=closest_patch_distance_to_prototype_j,
                )

                # add to the j-th heap
                if len(heaps[j]) < k:
                    heapq.heappush(heaps[j], closest_patch)
                else:
                    # heappushpop runs more efficiently than heappush
                    # followed by heappop
                    heapq.heappushpop(heaps[j], closest_patch)

    # after looping through the dataset every heap will
    # have the k closest prototypes
    for j in range(n_prototypes):
        # finally sort the heap; the heap only contains the k closest
        # but they are not ranked yet
        heaps[j].sort()
        heaps[j] = heaps[j][::-1]

    labels_all_prototype = np.array(
        [[patch.label for patch in heaps[j]] for j in range(n_prototypes)]
    )

    return labels_all_prototype


def prune_prototypes(
    dataloader,  # push loader
    model,
    k,
    prune_threshold,
):
    model.eval()
    ### run global analysis
    nearest_train_patch_class_ids = find_k_nearest_patches_to_prototypes(
        dataloader=dataloader,
        model=model,
        k=k,
    )

    prototypes_to_prune = []
    for j in range(model.num_prototypes()):
        # this will give us the class label of the prototype
        class_j = torch.argmax(model.prototype_class_identity[j]).item()

        if model.multi_label:
            # sum the binary label masks of the k closest patches
            nearest_train_patch_class_counts_j = np.sum(
                nearest_train_patch_class_ids[j], axis=0
            )
        else:
            nearest_train_patch_class_counts_j = Counter(
                nearest_train_patch_class_ids[j]
            )
        # if no such element is in Counter, it will return 0
        if nearest_train_patch_class_counts_j[class_j] < prune_threshold:
            prototypes_to_prune.append(j)

    ### bookkeeping of prototypes to be pruned
    class_of_prototypes_to_prune = (
        torch.argmax(model.prototype_class_identity[prototypes_to_prune], dim=1)
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    prototypes_to_prune_np = np.array(prototypes_to_prune).reshape(-1, 1)
    model.prune_prototypes(prototypes_to_prune)
