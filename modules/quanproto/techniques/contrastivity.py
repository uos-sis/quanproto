import torch
from torch.nn.functional import max_pool2d
from tqdm import tqdm

from quanproto.metrics import contrastivity, helpers


def evaluate_contrastivity(
    model,
    dataloader,
    num_prototypes_per_sample=5,
    metrics=[
        "vlc",
        "vac",
        "plc",
        "palc",
        "intra pd",
        "intra fd",
        "inter pd",
        "inter fd",
        "entropy",
        "histogram",
        "projection",
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

    if "palc" in metrics:
        total_palc = 0

    if "intra pd" in metrics:
        total_intra_prototype_distance = 0

    if "intra fd" in metrics:
        total_intra_feature_distance = 0

    if "inter pd" in metrics:
        total_prototypes = torch.empty(0).cuda()

    if "inter fd" in metrics:
        total_features = torch.empty(0).cuda()

    if "inter pd" in metrics or "inter fd" in metrics:
        total_one_hot_labels = torch.empty(0).cuda()

    if "projection" in metrics:
        total_labels = torch.empty(0)

    if "histogram" in metrics or "entropy" in metrics:
        total_prototype_indices = torch.empty(0).cuda()

    if "entropy" in metrics:
        total_similarities = torch.empty(0).cuda()

    if "projection" in metrics:
        total_prototype_activation = torch.empty(0).cuda()

    test_iter = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Contrastivity Evaluation",
        mininterval=2.0,
        ncols=0,
    )

    model.eval()
    for _, batch in test_iter:

        # region Check if the inputs are a list or not
        # INFO: This should be the same in all evaluation functions
        if isinstance(batch[0], (list, tuple)) and len(batch[0]) == 4:
            inputs = batch[0][0].cuda()
            explanation_inputs = batch[0][1].cuda()
            mask_bbox = batch[0][2]
            mask_size = batch[0][3]
        else:
            inputs = batch[0].cuda()
            explanation_inputs = None
        # endregion Check if the inputs are a list or not

        # region Evaluation specific inputs
        # INFO: tsne projection is possible in both the single label and multi label case. The inter
        # distances are only possible in the single label case because we do not know the class
        # labels of vectors in the multi label case
        if len(batch) == 2:
            if "projection" in metrics:
                total_labels = torch.cat([total_labels, batch[1]])

            if "inter pd" in metrics or "inter fd" in metrics:
                total_one_hot_labels = torch.cat(
                    [
                        total_one_hot_labels,
                        torch.nn.functional.one_hot(
                            batch[1].cuda(), num_classes=model.num_classes()
                        ),
                    ]
                )
        if len(batch) == 3:
            # save the animal class labels of the AWA2 dataset instead of the attribute labels to make a tsne plot for the prototype activations with the animal class labels
            # because the attribute labels do not make sense for the tsne plot
            if "projection" in metrics:
                total_labels = torch.cat([total_labels, batch[2]])
        # endregion Evaluation specific inputs

        with torch.no_grad():

            #
            # Get the similarity maps and saliency maps and the top k prototypes
            #

            _, similarity_maps, feature_map = model.explain(inputs)

            # get the maximum value of the similarity maps
            similarity_scores = (
                max_pool2d(similarity_maps, kernel_size=similarity_maps.shape[2:])
                .squeeze(-1)
                .squeeze(-1)
            )
            # get the top k similarity scores indices
            _, topk_indices = torch.topk(similarity_scores, k=k, dim=1)

            # use only the top k similarity maps
            similarity_maps = torch.stack(
                [similarity_maps[i, topk_indices[i]] for i in range(similarity_maps.shape[0])]
            )

            if "vlc" in metrics or "vac" in metrics:
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

                if use_bbox:
                    bbs = [
                        [helpers.bounding_box(saliency_maps[b, i]) for i in range(k)]
                        for b in range(saliency_maps.shape[0])
                    ]

            if "vlc" in metrics:
                if use_bbox:
                    total_vlc += torch.mean(contrastivity.intra_bb_location_change(bbs)).item()
                else:
                    total_vlc += torch.mean(
                        contrastivity.intra_mask_location_change(saliency_masks)
                    ).item()

            if "vac" in metrics:
                total_vac += torch.mean(
                    contrastivity.intra_mask_activation_change(saliency_maps)
                ).item()

            if "histogram" in metrics or "entropy" in metrics:
                total_prototype_indices = torch.cat(
                    [total_prototype_indices, topk_indices.view(-1)]
                )

            if "entropy" in metrics:
                total_similarities = torch.cat([total_similarities, similarity_scores])

            if "projection" in metrics:
                # make a one hot encoding of the top k indices
                topk_prototype_activation = torch.nn.functional.one_hot(
                    topk_indices, num_classes=model.num_prototypes()
                ).sum(dim=1)

                total_prototype_activation = torch.cat(
                    [total_prototype_activation, topk_prototype_activation]
                )

            if "inter pd" in metrics or "intra pd" in metrics:
                prototype_vectors = model.get_prototypes(topk_indices)
                if prototype_vectors == None:
                    raise ValueError(
                        "Prototype vectors are None, Model does not have explicit prototypes"
                    )
                prototype_vectors = prototype_vectors.squeeze()

            if "inter pd" in metrics:
                total_prototypes = torch.cat([total_prototypes, prototype_vectors])

            if "intra pd" in metrics:
                total_intra_prototype_distance += torch.mean(
                    contrastivity.intra_vector_distance(prototype_vectors)
                ).item()

            if "inter fd" in metrics or "intra fd" in metrics:
                feature_vectors = contrastivity.get_feature_vectors(similarity_maps, feature_map)

            if "inter fd" in metrics:
                total_features = torch.cat([total_features, feature_vectors])

            if "intra fd" in metrics:
                total_intra_feature_distance += torch.mean(
                    contrastivity.intra_vector_distance(feature_vectors)
                ).item()

            if "palc" in metrics:
                similarity_masks = torch.stack(
                    [
                        torch.stack(
                            [helpers.min_max_norm_mask(similarity_maps[b, i]) for i in range(k)]
                        )
                        for b in range(similarity_maps.shape[0])
                    ]
                )
                total_palc += torch.mean(
                    contrastivity.intra_mask_location_change(similarity_masks)
                ).item()

            if "plc" in metrics:
                total_plc += torch.mean(
                    contrastivity.max_activation_location_change(similarity_maps)
                ).item()

        del inputs, similarity_maps, feature_map

    if "vlc" in metrics:
        total_vlc /= len(dataloader)
        results["VLC"] = total_vlc * 100

    if "vac" in metrics:
        total_vac /= len(dataloader)
        results["VAC"] = total_vac * 100

    if "plc" in metrics:
        total_plc /= len(dataloader)
        results["PLC"] = total_plc

    if "palc" in metrics:
        total_palc /= len(dataloader)
        results["PALC"] = total_palc * 100

    if "intra pd" in metrics:
        total_intra_prototype_distance /= len(dataloader)
        results["Intra PD"] = total_intra_prototype_distance

    if "intra fd" in metrics:
        total_intra_feature_distance /= len(dataloader)
        results["Intra FD"] = total_intra_feature_distance

    if "inter pd" in metrics:
        total_inter_class_prototype_distance = torch.mean(
            contrastivity.inter_class_vector_distance(total_prototypes, total_one_hot_labels)
        ).item()
        results["Inter PD"] = total_inter_class_prototype_distance

    if "inter fd" in metrics:
        total_inter_class_feature_distance = torch.mean(
            contrastivity.inter_class_vector_distance(total_features, total_one_hot_labels)
        ).item()
        results["Inter FD"] = total_inter_class_feature_distance

    if "entropy" in metrics:
        total_prototype_entropy = contrastivity.activation_entropy(total_similarities)
        unique, _ = torch.unique(total_prototype_indices, return_counts=True)
        total_prototype_entropy = total_prototype_entropy[unique.int()]
        mean_entropy = torch.mean(total_prototype_entropy).item()
        results["entropy"] = mean_entropy

    if "histogram" in metrics:
        # make a histogram of the prototype indices
        unique, counts = torch.unique(total_prototype_indices, return_counts=True)
        prototype_histogram = dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))
        prototype_histogram = {str(int(k)): int(v) for k, v in prototype_histogram.items()}
        results["histogram"] = prototype_histogram

    # INFO: we can also make a tsne plot of the prototype activations
    if "projection" in metrics:
        projection = contrastivity.tsne_vector_projection(total_prototype_activation)
        prototype_projection = dict(zip(total_labels.cpu().numpy(), projection))
        prototype_projection = {
            str(int(k)): [float(v[0]), float(v[1])] for k, v in prototype_projection.items()
        }
        results["projection"] = prototype_projection

    # INFO: we wanted to test if another ml prediction method can be used to
    # predict the class labels, instead of the FullyConnected layer approach

    # # use a gaussian mixture model on the total_prototype_activation
    # from sklearn.mixture import GaussianMixture

    # gmm = GaussianMixture(n_components=model.num_classes(), random_state=42)
    # gmm.fit(total_prototype_activation.cpu().numpy())

    # TODO: we have to find a mapping from the gmm clusters to the class labels
    # This can be done with the hungarian algorithm
    # Reference: https://en.wikipedia.org/wiki/Hungarian_algorithm

    return results
