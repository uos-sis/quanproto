import torch
from torch.nn.functional import max_pool2d
from tqdm import tqdm

from quanproto.metrics import continuity, helpers, output_completeness


def evaluate_continuity(
    model,
    dataloader,
    num_prototypes_per_sample=5,
    metrics=[
        "vlc",
        "vac",
        "plc",
        "psc",
        "prc",
        "palc",
        "pac",
        "cac",
        "crc",
        "stability",
    ],
    use_bbox=False,
):

    k = num_prototypes_per_sample
    results = {}

    if "vlc" in metrics:
        total_vlc = 0

    if "vac" in metrics:
        total_vac = 0

    if "plc" in metrics:
        total_plc = 0

    if "psc" in metrics:
        total_psc = 0

    if "prc" in metrics:
        total_prc = 0

    if "palc" in metrics:
        total_palc = 0

    if "pac" in metrics:
        total_pac = 0

    if "cac" in metrics:
        total_cac = 0

    if "crc" in metrics:
        total_crc = 0

    if "stability" in metrics:
        total_stability = 0
        batch_count = 0.0

    test_iter = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Continuity Evaluation",
        mininterval=2.0,
        ncols=0,
    )

    model.eval()
    for _, (_inputs, new_inputs, partlocs, labels) in test_iter:

        # region Check if the inputs are a list or not
        # INFO: This should be the same in all evaluation functions
        if isinstance(_inputs, (list, tuple)) and len(_inputs) == 2:
            inputs = _inputs[0].cuda()
            explanation_inputs = _inputs[1].cuda()
        else:
            inputs = _inputs.cuda()
            explanation_inputs = None
        # endregion Check if the inputs are a list or not

        # region Evaluation specific inputs
        new_inputs = new_inputs.cuda()
        labels = labels.cuda()
        # endregion Evaluation specific inputs

        with torch.no_grad():
            #
            # Get the similarity maps and saliency maps and the top k prototypes
            #
            logits, similarity_maps, _ = model.explain(inputs)
            # get the maximum value of the similarity maps
            similarity_scores = (
                max_pool2d(similarity_maps, kernel_size=similarity_maps.shape[2:])
                .squeeze(-1)
                .squeeze(-1)
            )
            # get the top k similarity scores indices
            _, topk_indices = torch.topk(similarity_scores, k=k, dim=1)
            del similarity_scores

            # get the second input
            new_logits, new_similarity_maps, _ = model.explain(new_inputs)

            new_similarity_scores = (
                max_pool2d(
                    new_similarity_maps, kernel_size=new_similarity_maps.shape[2:]
                )
                .squeeze(-1)
                .squeeze(-1)
            )

            # region Process the similarity maps
            # INFO: remember the similarity maps are not normalized or clipped in any way
            # use only the top k similarity maps
            similarity_maps = torch.stack(
                [
                    similarity_maps[i, topk_indices[i]]
                    for i in range(similarity_maps.shape[0])
                ]
            )

            new_similarity_maps = torch.stack(
                [
                    new_similarity_maps[i, topk_indices[i]]
                    for i in range(new_similarity_maps.shape[0])
                ]
            )

            # INFO: we will use a min-max normalization to define the activation of a prototype in
            # the similarity map as stated in the paper
            similarity_masks = torch.stack(
                [
                    torch.stack(
                        [
                            helpers.min_max_norm_mask(similarity_maps[b, i])
                            for i in range(k)
                        ]
                    )
                    for b in range(similarity_maps.shape[0])
                ]
            )

            new_similarity_masks = torch.stack(
                [
                    torch.stack(
                        [
                            helpers.min_max_norm_mask(new_similarity_maps[b, i])
                            for i in range(k)
                        ]
                    )
                    for b in range(new_similarity_maps.shape[0])
                ]
            )
            # endregion Process the similarity maps

            # region Compute the saliency maps
            # INFO: We only compute the saliency maps if the metrics require it, because some
            # saliency methods are computationally expensive
            if "vlc" in metrics or "vac" in metrics or "stability" in metrics:
                # INFO: The explanation_inputs variable is a way to provide extra input to the
                # saliency technique. This implementation is quite specific, because the saliency map is not
                # clipped afterwards and directly used as a mask. We use this hotfix to provide
                # segmentation masks to the saliency technique.
                if explanation_inputs is not None:
                    saliency_maps = model.saliency_maps(
                        inputs, explanation_inputs, topk_indices
                    )
                    saliency_masks = saliency_maps.clone().detach()
                else:
                    # INFO: This is the default way to compute the saliency maps, which is used for
                    # ProtoPNet, ProtoPool, PIPNet, ProtoTree
                    saliency_maps = model.saliency_maps(inputs, topk_indices)
                    # get a clipping mask for the saliency maps
                    saliency_masks = torch.stack(
                        [
                            torch.stack(
                                [
                                    helpers.percentile_mask(saliency_maps[b, i])
                                    for i in range(k)
                                ]
                            )
                            for b in range(saliency_maps.shape[0])
                        ]
                    )
                    # clip the saliency maps
                    saliency_maps = saliency_maps * saliency_masks

                if explanation_inputs is not None:
                    new_saliency_maps = model.saliency_maps(
                        new_inputs, explanation_inputs, topk_indices
                    )
                    new_saliency_masks = new_saliency_maps.clone().detach()
                else:
                    new_saliency_maps = model.saliency_maps(new_inputs, topk_indices)
                    new_saliency_masks = torch.stack(
                        [
                            torch.stack(
                                [
                                    helpers.percentile_mask(new_saliency_maps[b, i])
                                    for i in range(k)
                                ]
                            )
                            for b in range(new_saliency_maps.shape[0])
                        ]
                    )
                    new_saliency_maps = new_saliency_maps * new_saliency_masks

                bbs = [
                    [helpers.bounding_box(saliency_maps[b, i]) for i in range(k)]
                    for b in range(saliency_maps.shape[0])
                ]

                new_bbs = [
                    [helpers.bounding_box(new_saliency_maps[b, i]) for i in range(k)]
                    for b in range(new_saliency_maps.shape[0])
                ]

            if "vlc" in metrics:
                if use_bbox:
                    total_vlc += torch.mean(
                        output_completeness.bb_location_change(bbs, new_bbs)
                    ).item()
                else:
                    total_vlc += torch.mean(
                        output_completeness.mask_location_change(
                            saliency_masks, new_saliency_masks
                        )
                    ).item()

            if "vac" in metrics:
                total_vac += torch.mean(
                    output_completeness.activation_change(
                        saliency_maps, new_saliency_maps
                    )
                ).item()

            if "plc" in metrics:
                total_plc += torch.mean(
                    output_completeness.max_activation_location_change(
                        similarity_maps, new_similarity_maps
                    ).float()
                ).item()

            if "psc" in metrics:
                total_psc += torch.mean(
                    output_completeness.max_activation_change(
                        similarity_maps, new_similarity_maps
                    )
                ).item()

            if "prc" in metrics:
                total_prc += torch.mean(
                    output_completeness.rank_change(
                        topk_indices, new_similarity_scores
                    ).float()
                ).item()

            if "palc" in metrics:
                total_palc += torch.mean(
                    output_completeness.mask_location_change(
                        similarity_masks, new_similarity_masks
                    )
                ).item()

            if "pac" in metrics:
                total_pac += torch.mean(
                    output_completeness.activation_change(
                        similarity_maps, new_similarity_maps
                    )
                ).item()

            if "cac" in metrics:
                total_cac += torch.mean(
                    continuity.classification_activation_change(logits, new_logits)
                ).item()

            if "crc" in metrics and model.multi_label is False:
                total_crc += torch.mean(
                    continuity.classification_rank_change(logits, new_logits).float()
                ).item()

            if "stability" in metrics:
                if partlocs.shape[1] == 0:
                    raise NotImplementedError(
                        "The stability metric requires partlocs to be provided"
                    )

                if use_bbox:
                    stability = continuity.stability_score_bb(bbs, new_bbs, partlocs)
                else:
                    stability = continuity.stability_score_mask(
                        saliency_masks, new_saliency_masks, partlocs
                    )

                # remove the nan values
                # nan values are generated when the saliency masks does not overlap with any
                # partlocs
                stability = stability[~torch.isnan(stability)]

                # if every saliency mask does not overlap with any partlocs, we will not count it
                # in the mean
                if stability.numel() != 0:
                    total_stability += torch.mean(stability).item()
                    batch_count += 1

        del (
            inputs,
            new_inputs,
            labels,
            logits,
            new_logits,
            similarity_maps,
            new_similarity_maps,
        )

    if "vlc" in metrics:
        results["VLC"] = (total_vlc / len(dataloader)) * 100

    if "vac" in metrics:
        results["VAC"] = (total_vac / len(dataloader)) * 100

    if "plc" in metrics:
        results["PLC"] = total_plc / len(dataloader)

    if "psc" in metrics:
        results["PSC"] = (total_psc / len(dataloader)) * 100

    if "prc" in metrics:
        results["PRC"] = total_prc / len(dataloader)

    if "palc" in metrics:
        results["PALC"] = (total_palc / len(dataloader)) * 100

    if "pac" in metrics:
        results["PAC"] = (total_pac / len(dataloader)) * 100

    if "cac" in metrics:
        results["CAC"] = (total_cac / len(dataloader)) * 100

    if "crc" in metrics:
        results["CRC"] = total_crc / len(dataloader)

    if "stability" in metrics:
        results["Stability"] = (total_stability / batch_count) * 100

    return results
