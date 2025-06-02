import torch


def intra_mask_location_change(map_batch: torch.Tensor):
    """
    Compute 1 - intersection over union between masks

    Args:
        map_batch: The first batch of activation maps (B x N x H x W)
    """
    # expand the tensor to Bx1xNxHxW
    map_batch1 = map_batch.unsqueeze(1)
    # expand the tensor to BxNx1xHxW
    map_batch2 = map_batch.unsqueeze(2)

    # reshape the tensors to BxNxNxHxW
    map_batch1 = map_batch1.repeat(1, map_batch.shape[1], 1, 1, 1)
    map_batch2 = map_batch2.repeat(1, 1, map_batch.shape[1], 1, 1)

    # calculate the intersection
    intersection = torch.logical_and(map_batch1, map_batch2).sum(dim=(3, 4))
    union = torch.logical_or(map_batch1, map_batch2).sum(dim=(3, 4))

    overlap = intersection / union
    change = 1 - overlap

    # calculate the mean change of a map to all other maps
    # the -1 is important to not consider the map itself
    change = change.sum(dim=2) / (map_batch.shape[1] - 1)
    change = change.mean(dim=1)

    return change


def intra_bb_location_change(bb_batch: torch.Tensor):
    """
    Compute the bounding box of the maps
    Compute 1 - intersection over union

    Args:
        bb_batch: The first batch of bounding boxes (B x N x 4)
    """
    if not isinstance(bb_batch, torch.Tensor):
        bb_batch = torch.tensor(bb_batch)
    B, N, _ = bb_batch.size()

    # Prepare an output tensor for intersections
    intersections = torch.zeros((B, N, N, 4), dtype=bb_batch.dtype, device=bb_batch.device)

    # Expand to compute intersections for all pairs
    # bb_batch_a and bb_batch_b represent pairs of bb_batch
    bb_batch_a = bb_batch.unsqueeze(2)  # Shape: (B, N, 1, 4)
    bb_batch_b = bb_batch.unsqueeze(1)  # Shape: (B, 1, N, 4)

    # Compute intersection bounds
    intersect_lower_x = torch.maximum(bb_batch_a[..., 0], bb_batch_b[..., 0])  # max(lower_x)
    intersect_lower_y = torch.maximum(bb_batch_a[..., 1], bb_batch_b[..., 1])  # max(lower_y)
    intersect_upper_x = torch.minimum(bb_batch_a[..., 2], bb_batch_b[..., 2])  # min(upper_x)
    intersect_upper_y = torch.minimum(bb_batch_a[..., 3], bb_batch_b[..., 3])  # min(upper_y)

    # Condition for valid intersections
    valid_intersections = (intersect_lower_x < intersect_upper_x) & (
        intersect_lower_y < intersect_upper_y
    )

    # Store in the output tensor, where valid intersections are computed
    intersections[..., 0] = torch.where(valid_intersections, intersect_lower_x, 0)
    intersections[..., 1] = torch.where(valid_intersections, intersect_lower_y, 0)
    intersections[..., 2] = torch.where(valid_intersections, intersect_upper_x, 0)
    intersections[..., 3] = torch.where(valid_intersections, intersect_upper_y, 0)

    intersections_area = (intersections[..., 2] - intersections[..., 0]) * (
        intersections[..., 3] - intersections[..., 1]
    )  # Shape: (B, N, N)

    # Compute the area of the bounding boxes
    bb_area = (bb_batch[..., 2] - bb_batch[..., 0]) * (
        bb_batch[..., 3] - bb_batch[..., 1]
    )  # Shape: (B, N)

    # Compute the union area
    union_area = bb_area.unsqueeze(2) + bb_area.unsqueeze(1) - intersections_area

    # Compute the overlap
    overlap = intersections_area / union_area
    change = 1 - overlap
    change = change.sum(dim=2) / (N - 1)
    change = change.mean(dim=1)
    return change


def intra_mask_activation_change(map_batch: torch.Tensor):
    """

    Args:
        map_batch: The first batch of activation maps (B x N x H x W)
    """
    # flatten the activation maps to (BxNx H*W)
    flat_map1 = map_batch.view(map_batch.shape[0], map_batch.shape[1], -1)
    # expand the tensor to Bx1xNxH*W
    flat_map1 = flat_map1.unsqueeze(1)

    # flatten the activation maps to (BxNx H*W)
    flat_map2 = map_batch.view(map_batch.shape[0], map_batch.shape[1], -1)
    # expand the tensor to BxNx1xH*W
    flat_map2 = flat_map2.unsqueeze(2)

    # sort the activation maps
    flat_map1, _ = torch.sort(flat_map1, dim=3, descending=True)
    flat_map2, _ = torch.sort(flat_map2, dim=3, descending=True)

    # reshape the tensors to BxNxNxH*W
    flat_map1 = flat_map1.repeat(1, map_batch.shape[1], 1, 1)
    flat_map2 = flat_map2.repeat(1, 1, map_batch.shape[1], 1)

    # calculate the intersection
    intersection = torch.minimum(flat_map1, flat_map2).sum(dim=3)
    union = torch.maximum(flat_map1, flat_map2).sum(dim=3)

    # check if nan values are present
    assert not torch.any(torch.isnan(intersection)), "Intersection contains NaN values"
    assert not torch.any(torch.isnan(union)), "Union contains NaN values"

    overlap = intersection / union

    assert not torch.any(torch.isnan(overlap)), "Overlap contains NaN values"

    change = 1 - overlap

    # calculate the mean change of a map to all other maps
    # the -1 is important to not consider the map itself
    change = change.sum(dim=2) / (map_batch.shape[1] - 1)
    change = change.mean(dim=1)

    return change


def max_activation_location_change(map_batch: torch.Tensor):
    """

    Args:
        map_batch: The first batch of activation maps (B x N x H x W)
    """
    # flatten the activation maps to (BxNx H*W)
    flat_map = map_batch.view(map_batch.shape[0], map_batch.shape[1], -1)

    # find the maximum activations
    max_map1 = torch.argmax(flat_map, dim=2)

    # unravel the index to get the coordinates BxNx2
    max_map1 = torch.stack([max_map1 // map_batch.shape[2], max_map1 % map_batch.shape[2]], dim=2)

    # expand the tensor to Bx1xNx2
    map1 = max_map1.unsqueeze(1)
    # expand the tensor to BxNx1x2
    map2 = max_map1.unsqueeze(2)

    # reshape the tensors to BxNxNx2
    map1 = map1.repeat(1, map_batch.shape[1], 1, 1)
    map2 = map2.repeat(1, 1, map_batch.shape[1], 1)

    # compute the manhatten distance
    distance = torch.abs(map1 - map2).sum(dim=3)
    distance = distance.sum(dim=2)
    distance = distance / (map_batch.shape[1] - 1)
    distance = distance.mean(dim=1)
    return distance


def get_feature_vectors(map_batch: torch.Tensor, feature_map: torch.Tensor):
    """
    Find the feature vectors of the top k activation maps.

    Args:
        map_batch: The first batch of activation maps (B x N x H x W)
        feature_map: The feature map of the model (B x D x H x W)
    """

    # flatten the activation maps to (BxNx H*W)
    flat_map = map_batch.view(map_batch.shape[0], map_batch.shape[1], -1)

    # flatten the feature map to (BxDx H*W)
    flat_feature = feature_map.view(feature_map.shape[0], feature_map.shape[1], -1)

    # find the maximum activations
    max_map = torch.argmax(flat_map, dim=2)

    # get the feature vectors of the top k activation maps
    features = torch.stack([flat_feature[i, :, max_map[i]] for i in range(flat_feature.shape[0])])

    # switch dimension to BxNxD
    features = features.permute(0, 2, 1)
    return features


def intra_vector_distance(vectors: torch.Tensor) -> float:
    """
    The average cosine distance between vectors of the same classes.

    Args:
        vectors: The vectors to calculate the distance between. (B x N x D)
    """
    # unsqueeze the vectors to Bx1xNxD
    vectors1 = vectors.unsqueeze(1)
    # unsqueeze the vectors to BxNx1xD
    vectors2 = vectors.unsqueeze(2)

    # reshape the tensors to BxNxNxD
    vectors1 = vectors1.repeat(1, vectors.shape[1], 1, 1)
    vectors2 = vectors2.repeat(1, 1, vectors.shape[1], 1)

    # calculate the cosine similarity
    similarity = torch.nn.functional.cosine_similarity(vectors1, vectors2, dim=3)

    # calculate the distance
    distance = 1 - similarity
    # all distances smaller 0.e-5 are set to 0
    distance = torch.where(distance < 0.00001, torch.zeros_like(distance), distance)

    # calculate the mean distance of a vector to all other vectors
    # the -1 is important to not consider the vector itself
    distance = distance.sum(dim=2) / (vectors.shape[1] - 1)

    distance = distance.mean(dim=1)

    return distance


def inter_class_vector_distance(vectors: torch.Tensor, class_identity: torch.Tensor):
    """
    The average cosine distance between vectors of different classes.

    Args:
        vectors: The vectors to calculate the distance between. (B x N x D)
        class_identity: The class identity of each vector. (B x Classes)
    """
    # get the class ids present in the batch
    class_id_mask = torch.sum(class_identity, dim=0) > 0

    class_ids = torch.nonzero(class_id_mask).squeeze()

    inter_class_distance = torch.zeros(class_identity.shape[1])

    # if only one class is present
    if class_ids.numel() == 1:
        return inter_class_distance

    for class_id in class_ids:
        # get the vectors of the class Z x N x D
        class_vectors = vectors[class_identity[:, class_id] == 1]
        # get the vectors of all other classes Y x N x D
        other_vectors = vectors[class_identity[:, class_id] == 0]

        # TODO: this allocates too much memory
        # # expand the tensor to Zx1xNxD
        # class_vectors = class_vectors.unsqueeze(1)
        # # expand the tensor to 1xYxNxD
        # other_vectors = other_vectors.unsqueeze(0)

        # # reshape the tensors to ZxYxNxD
        # class_vectors = class_vectors.repeat(1, other_vectors.shape[1], 1, 1)
        # other_vectors = other_vectors.repeat(class_vectors.shape[0], 1, 1, 1)

        # # calculate the cosine similarity
        # similarity = torch.nn.functional.cosine_similarity(class_vectors, other_vectors, dim=3)
        # distance = 1 - similarity

        # # calculate the mean distance of a vector to all other vectors
        # class_distance = torch.mean(distance).item()

        # # save the distance
        # inter_class_distance[class_id] = class_distance

        total_inter_dist = 0
        for i in range(class_vectors.shape[0]):
            class_vec = class_vectors[i].unsqueeze(0)
            similarity = torch.nn.functional.cosine_similarity(class_vec, other_vectors, dim=2)
            distance = 1 - similarity
            total_inter_dist += distance.mean().item()

        total_inter_dist /= class_vectors.shape[0]
        inter_class_distance[class_id] = total_inter_dist

    return inter_class_distance


def tsne_vector_projection(
    vectors: torch.Tensor, perplexity: int = 30, n_iter: int = 1000
) -> torch.Tensor:
    """
    The t-SNE projection of the vectors.

    Args:
        vectors: The vectors to project. (N x D)
        perplexity: The perplexity parameter of t-SNE.
        n_iter: The number of iterations to run t-SNE.
    """
    from sklearn.manifold import TSNE

    vectors = vectors.squeeze()

    tsne = TSNE(perplexity=perplexity, n_iter=n_iter, n_jobs=-1)
    projection = tsne.fit_transform(vectors.cpu().detach().numpy())
    return projection


def activation_entropy(similarity_scores: torch.Tensor, bins=10):
    """
    Compute the entropy of the prototype activation

    Args:
        similarity_scores: The similarity scores (B x N)
    """
    # max normalization
    norm_scores = similarity_scores / torch.max(similarity_scores, dim=0, keepdim=True)[0]

    # make a histogram of the similarity scores over dimension 1
    histograms = torch.zeros(similarity_scores.shape[1], bins).to(similarity_scores.device)

    for i in range(similarity_scores.shape[1]):
        histograms[i] = torch.histc(norm_scores[:, i], bins=bins, min=0.0, max=1.0)

    # make probability values from the histrograms
    probs = histograms / torch.sum(histograms, dim=1, keepdim=True)

    # check if the sum of the probabilities is 1
    assert torch.allclose(torch.sum(probs, dim=1), torch.ones(probs.shape[0]).to(probs.device))

    # TODO: this is numerically unstable, so we get entropy values > 1
    # every prob below 0.1% is set to 0.0% to avoid numerical instability
    probs = torch.where(probs < 0.001, torch.zeros_like(probs), probs)

    prob_logs = probs * torch.log2(probs)
    prob_logs[torch.isnan(prob_logs)] = 0
    entropy = -torch.sum(prob_logs, dim=1)
    entropy = torch.clamp(entropy, 0, 1)

    return entropy
