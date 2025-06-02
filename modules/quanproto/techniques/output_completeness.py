import torch
from torch.nn.functional import max_pool2d
from tqdm import tqdm

from quanproto.metrics import helpers, output_completeness


def evaluate_output_completeness(
    model,
    dataloader,
    num_prototypes_per_sample=5,
    std=0.05,
    metrics=["vlc", "vac", "plc", "psc", "prc", "palc", "pac"],
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

    # total_prototype_indices = torch.empty(0).cuda()

    test_iter = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Output-Completeness Evaluation",
        mininterval=2.0,
        ncols=0,
    )

    model.eval()
    for _, (_inputs, labels) in test_iter:
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
        labels = labels.cuda()
        # endregion Evaluation specific inputs

        with torch.no_grad():
            # region TopK Prototypes ---------------------------------------------------------------
            _, similarity_maps, _ = model.explain(inputs)

            # get the maximum value of the similarity maps
            similarity_scores = (
                max_pool2d(similarity_maps, kernel_size=similarity_maps.shape[2:])
                .squeeze(-1)
                .squeeze(-1)
            )
            # get the top k similarity scores indices
            _, topk_indices = torch.topk(similarity_scores, k=k, dim=1)
            # endregion TopK Prototypes ------------------------------------------------------------

            # total_prototype_indices = torch.cat(
            #     [total_prototype_indices, topk_indices.view(-1)]
            # )

            # region Similarity Maps ---------------------------------------------------------------
            # use only the top k saliency maps
            similarity_maps = torch.stack(
                [
                    similarity_maps[i, topk_indices[i]]
                    for i in range(similarity_maps.shape[0])
                ]
            )
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
            # endregion Similarity Maps ------------------------------------------------------------

            # region Saliency Maps -----------------------------------------------------------------
            if explanation_inputs is not None:
                saliency_maps = model.saliency_maps(
                    inputs, explanation_inputs, topk_indices
                )
                saliency_masks = saliency_maps.clone().detach()
            else:
                saliency_maps = model.saliency_maps(inputs, topk_indices)
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
                saliency_maps = saliency_maps * saliency_masks

            bbs = [
                [helpers.bounding_box(saliency_maps[b, i]) for i in range(k)]
                for b in range(saliency_maps.shape[0])
            ]
            # endregion Saliency Maps ------------------------------------------------------------

            if (
                "plc" in metrics
                or "palc" in metrics
                or "psc" in metrics
                or "pac" in metrics
            ):
                new_similarity_maps = torch.empty(0).cuda()
                new_similarity_masks = torch.empty(0).cuda()

            if "prc" in metrics:
                new_similarity_scores = torch.empty(0).cuda()

            if "vlc" in metrics or "vac" in metrics:
                new_saliency_maps = torch.empty(0).cuda()
                new_saliency_masks = torch.empty(0).cuda()

            #
            # Perturb the input images based on the prototype saliency maps
            #
            for idx in range(k):
                # copy images and add noise based on topk idx
                new_inputs = inputs.clone()
                new_inputs = torch.stack(
                    [
                        output_completeness.add_gaussian_noise(
                            new_inputs[b], bbs[b][idx], std=std
                        )
                        for b in range(new_inputs.shape[0])
                    ]
                )

                _, similarity_maps_perturbed, _ = model.explain(new_inputs)

                if "prc" in metrics:
                    similarity_scores_perturbed = (
                        max_pool2d(
                            similarity_maps_perturbed,
                            kernel_size=similarity_maps_perturbed.shape[2:],
                        )
                        .squeeze(-1)
                        .squeeze(-1)
                    )
                    new_similarity_scores = torch.cat(
                        [
                            new_similarity_scores,
                            similarity_scores_perturbed.unsqueeze(1),
                        ],
                        dim=1,
                    )

                if (
                    "plc" in metrics
                    or "palc" in metrics
                    or "psc" in metrics
                    or "pac" in metrics
                ):
                    # use only the similarity map from the current prototype idx
                    similarity_maps_perturbed = torch.stack(
                        [
                            similarity_maps_perturbed[i, topk_indices[i, idx]]
                            for i in range(similarity_maps_perturbed.shape[0])
                        ]
                    ).unsqueeze(1)
                    similarity_masks_perturbed = helpers.min_max_norm_mask(
                        similarity_maps_perturbed
                    )
                    new_similarity_masks = torch.cat(
                        [new_similarity_masks, similarity_masks_perturbed], dim=1
                    )
                    # save the similarity maps
                    new_similarity_maps = torch.cat(
                        [new_similarity_maps, similarity_maps_perturbed], dim=1
                    )

                if "vlc" in metrics or "vac" in metrics:
                    if explanation_inputs is not None:
                        saliency_maps_perturbed = model.saliency_maps(
                            new_inputs,
                            explanation_inputs,
                            topk_indices[:, idx].unsqueeze(1),
                        )
                        saliency_masks_perturbed = (
                            saliency_maps_perturbed.clone().detach()
                        )
                    else:
                        saliency_maps_perturbed = model.saliency_maps(
                            new_inputs, topk_indices[:, idx].unsqueeze(1)
                        )
                        saliency_masks_perturbed = helpers.percentile_mask(
                            saliency_maps_perturbed
                        )

                        saliency_maps_perturbed = (
                            saliency_maps_perturbed * saliency_masks_perturbed
                        )

                    # save the saliency maps
                    new_saliency_maps = torch.cat(
                        [new_saliency_maps, saliency_maps_perturbed], dim=1
                    )

                    new_saliency_masks = torch.cat(
                        [new_saliency_masks, saliency_masks_perturbed], dim=1
                    )

                del (
                    new_inputs,
                    similarity_maps_perturbed,
                )

            if "vlc" in metrics:
                if use_bbox:
                    new_bbs = [
                        [
                            helpers.bounding_box(new_saliency_maps[b, i])
                            for i in range(k)
                        ]
                        for b in range(new_saliency_maps.shape[0])
                    ]
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

        del (
            inputs,
            labels,
            similarity_scores,
            similarity_maps,
            saliency_maps,
            topk_indices,
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

    return results
