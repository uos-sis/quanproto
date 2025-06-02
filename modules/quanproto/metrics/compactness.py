import torch


def local_explanation_size(similarity_scores: torch.Tensor, threshold=0.1) -> torch.Tensor:
    """compute the size of the explanation as the number of non-zero activations

    :param similarity_scores: vector of similarity scores (batch_size x num_prototypes)
    :type similarity_scores: torch.Tensor
    :param threshold: minimal normalized percentile to consider the prototype activated, defaults to 0.1
    :type threshold: float, optional
    :return: explanation size (batch_size)
    :rtype: torch.Tensor
    """
    # max normalization
    norm_scores = similarity_scores / torch.max(similarity_scores, dim=1, keepdim=True)[0]

    # count the number of non-zero activations
    explanation_size = torch.sum(norm_scores > threshold, axis=1).float()

    return explanation_size
