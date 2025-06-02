import torch
from torch.nn.functional import max_pool2d
from tqdm import tqdm

from quanproto.metrics import compactness


def evaluate_compactness(
    model,
    dataloader,
    local_size_threshold=0.1,
    metrics=["global size", "sparsity", "npr", "local size"],
):

    epsilon = 1e-3
    results = {}

    if "global size" in metrics:
        results["Global Size"] = model.global_explanation_size(epsilon)

    if "sparsity" in metrics:
        results["Sparsity"] = model.classification_sparsity(epsilon) * 100

    if "npr" in metrics:
        results["NPR"] = model.negative_positive_reasoning_ratio(epsilon)

    if "local size" in metrics:
        total_local_explanation_size = 0

        test_iter = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Compactness Evaluation",
            mininterval=2.0,
            ncols=0,
        )

        model.eval()
        for _, (inputs, _) in test_iter:
            inputs = inputs.cuda()

            with torch.no_grad():

                _, similarity_maps, _ = model.explain(inputs)

                # get the maximum value of the similarity maps
                similarity_scores = (
                    max_pool2d(similarity_maps, kernel_size=similarity_maps.shape[2:])
                    .squeeze(-1)
                    .squeeze(-1)
                )

                local_explanation_size = compactness.local_explanation_size(
                    similarity_scores, threshold=local_size_threshold
                )
                total_local_explanation_size += torch.mean(
                    local_explanation_size
                ).item()

            del inputs, similarity_maps

        results["Local Size"] = total_local_explanation_size / len(dataloader)

    return results
