import torch
from torch.nn.functional import max_pool2d
from tqdm import tqdm

from quanproto.metrics import complexity, helpers


def evaluate_complexity(
    model,
    dataloader,
    num_prototypes_per_sample=5,
    metrics=[
        "ior",
        "oirr",
        "object overlap",
        "iou",
        "background overlap",
        "consistency",
    ],
    use_bbox=False,
):

    k = num_prototypes_per_sample
    results = {}

    if "object overlap" in metrics:
        total_mask_overlap = 0

    if "oirr" in metrics:
        total_oirr = 0

    if "ior" in metrics:
        total_ior = 0

    if "iou" in metrics:
        total_iou = 0

    if "background overlap" in metrics:
        total_background_overlap = 0

    if "consistency" in metrics:
        total_prototype_indices = torch.empty(0).cuda()
        total_prototype_partlocs_ids = []
        for i in range(model.num_prototypes()):
            total_prototype_partlocs_ids.append(torch.empty(0).cuda())

    test_iter = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Complexity Evaluation",
        mininterval=2.0,
        ncols=0,
    )

    model.eval()
    for _, (_inputs, seg_masks, partlocs, partlocs_ids, labels) in test_iter:

        # region Check if the inputs are a list or not
        # INFO: This should be the same in all evaluation functions
        if isinstance(_inputs, (list, tuple)) and len(_inputs) == 4:
            inputs = _inputs[0].cuda()
            explanation_inputs = _inputs[1].cuda()
            mask_bbox = _inputs[2]
            mask_size = _inputs[3]
        else:
            inputs = _inputs.cuda()
            explanation_inputs = None
        # endregion Check if the inputs are a list or not

        # region Evaluation specific inputs
        seg_masks = seg_masks.cuda()
        partlocs = partlocs.cuda()
        partlocs_ids = partlocs_ids.cuda()
        labels = labels.cuda()
        # endregion Evaluation specific inputs

        with torch.no_grad():

            # region TopK Prototypes ---------------------------------------------------------------
            _, similarity_maps, _ = model.explain(inputs)

            similarity_scores = (
                max_pool2d(similarity_maps, kernel_size=similarity_maps.shape[2:])
                .squeeze(-1)
                .squeeze(-1)
            )
            # get the top k similarity scores indices
            _, topk_indices = torch.topk(similarity_scores, k=k, dim=1)
            # endregion TopK Prototypes ------------------------------------------------------------

            # region Saliency Maps -----------------------------------------------------------------
            # INFO: The explanation_inputs variable is a way to provide extra input to the
            # saliency technique. This implementation is quite specific, because the saliency map is not
            # clipped afterwards and directly used as a mask. We use this hotfix to provide
            # segmentation masks to the saliency technique.
            if explanation_inputs is not None:
                saliency_maps = model.saliency_maps(
                    inputs, explanation_inputs, mask_bbox, mask_size, topk_indices
                )
                if model.explanation_type == "mask":
                    saliency_masks = saliency_maps.clone().detach()
                if model.explanation_type == "prp":
                    saliency_masks = torch.stack(
                        [
                            torch.stack(
                                [helpers.percentile_mask(saliency_maps[b, i]) for i in range(k)]
                            )
                            for b in range(saliency_maps.shape[0])
                        ]
                    )
                    saliency_maps = saliency_maps * saliency_masks
            else:
                # INFO: This is the default way to compute the saliency maps, which is used for
                # ProtoPNet, ProtoPool, PIPNet, ProtoTree
                saliency_maps = model.saliency_maps(inputs, topk_indices)
                saliency_masks = torch.stack(
                    [
                        torch.stack(
                            [helpers.percentile_mask(saliency_maps[b, i]) for i in range(k)]
                        )
                        for b in range(saliency_maps.shape[0])
                    ]
                )
                saliency_maps = saliency_maps * saliency_masks
            # endregion Saliency Maps ------------------------------------------------------------

            # convert the sag mask into the needed format.
            if seg_masks.shape[3] == 3:  # if loaded with torch
                seg_masks = torch.sum(seg_masks, dim=3)  # B x H x W x C -> B x H x W
            if seg_masks.shape[1] == 3:  # if loaded with PIL
                seg_masks = torch.sum(seg_masks, dim=1)  # B x C x H x W -> B x H x W

            # INFO: use a min max normalization the mask to use only the certain pixels. This is
            # done as the CUB200 segmentation masks often have uncertain tree parts like the stick
            # on which the bird is sitting. This is not part of the bird and should not be used. We
            # hope that this normalization clipping is enough to remove these parts, but did not
            # test it.
            seg_masks = torch.stack(
                [helpers.min_max_norm_mask(seg_masks[b]) for b in range(seg_masks.shape[0])]
            )

            if "object overlap" in metrics:
                total_mask_overlap += torch.mean(
                    complexity.mask_overlap(saliency_masks, seg_masks)
                ).item()

            if "ior" in metrics:
                total_ior += torch.mean(
                    complexity.inside_outside_relevance(saliency_maps, seg_masks)
                ).item()

            if "oirr" in metrics:
                total_oirr += torch.mean(
                    complexity.outside_inside_relevance_ratio(saliency_maps, seg_masks)
                ).item()

            if "iou" in metrics:
                total_iou += torch.mean(
                    complexity.mask_intersection_over_union(saliency_masks, seg_masks)
                ).item()

            if "background overlap" in metrics:
                total_background_overlap += torch.mean(
                    complexity.background_overlap(saliency_masks, seg_masks)
                ).item()

            if "consistency" in metrics:
                total_prototype_indices = torch.cat(
                    [total_prototype_indices, topk_indices.view(-1)]
                )
                if partlocs.shape[1] == 0:
                    raise NotImplementedError(
                        "The consistency metric requires partlocs to be provided"
                    )

                if use_bbox:
                    bbs = [
                        [helpers.bounding_box(saliency_maps[b, i]) for i in range(k)]
                        for b in range(saliency_maps.shape[0])
                    ]

                    prototype_partloc_ids = complexity.boundingbox_consistency(
                        bbs, partlocs, partlocs_ids.squeeze()
                    )
                else:
                    prototype_partloc_ids = complexity.map_consistency(
                        saliency_maps, partlocs, partlocs_ids.squeeze()
                    )

                # get non zero indices
                for b in range(prototype_partloc_ids.shape[0]):
                    for i in range(k):
                        non_zero = prototype_partloc_ids[b, i].nonzero()

                        if non_zero.shape[0] == 0:
                            continue

                        non_zero_indices = prototype_partloc_ids[b, i][non_zero]

                        # INFO: The consistency metric only works if partlocs_ids start at 1 so
                        # 0 can be used as not assigned
                        # INFO We use the partloc id as an extra index array and not the indexing of
                        # the partloc array itself, because partlocs are saved sparse in our dataset
                        # implementation

                        # add the non zero indices to the prototype partlocs ids
                        prototype_id = topk_indices[b, i]
                        total_prototype_partlocs_ids[prototype_id] = torch.cat(
                            [
                                total_prototype_partlocs_ids[prototype_id],
                                non_zero_indices,
                            ]
                        )

        del inputs, labels, similarity_maps, saliency_maps, saliency_masks

    if "object overlap" in metrics:
        results["Object Overlap"] = (total_mask_overlap / len(dataloader)) * 100

    if "background overlap" in metrics:
        results["Background Overlap"] = (total_background_overlap / len(dataloader)) * 100

    if "ior" in metrics:
        results["IOR"] = total_ior / len(dataloader)

    if "oirr" in metrics:
        results["OIRR"] = total_oirr / len(dataloader)

    if "iou" in metrics:
        results["IOU"] = (total_iou / len(dataloader)) * 100

    if "consistency" in metrics:
        # make a histogram of the prototype indices
        unique_prototype_ids, counts_prototype_ids = torch.unique(
            total_prototype_indices, return_counts=True
        )
        prototype_histogram = dict(
            zip(unique_prototype_ids.cpu().numpy(), counts_prototype_ids.cpu().numpy())
        )
        prototype_histogram = {str(k): int(v) for k, v in prototype_histogram.items()}

        # convert the partlocs ids to coverage distributions
        total_coverage = 0
        for i in range(len(total_prototype_partlocs_ids)):
            if total_prototype_partlocs_ids[i].shape[0] == 0:
                continue

            _, counts = torch.unique(total_prototype_partlocs_ids[i], return_counts=True)

            # compute the coverage of the partlocs by dividing the counts with the
            coverage_distribution = counts / prototype_histogram[str(float(i))]
            total_coverage += torch.mean(coverage_distribution).item()
        total_coverage /= unique_prototype_ids.shape[0]
        results["Consistency"] = total_coverage * 100

    return results
