import torch
from torch import nn

from quanproto.functional.activation import LogActivation
from quanproto.functional.l2conv import L2SquaredConv2d
from quanproto.functional.minpool import min_pool2d
from quanproto.models.helper import (
    compute_mean_weight,
    convert_to_multilabelmargin_input,
    model_output_size,
)
from quanproto.models.interfaces.prototype_model_interfaces import (
    PrototypeModelInterface,
)


class ProtoMask(nn.Module, PrototypeModelInterface):

    def __init__(
        self,
        backbone: nn.Module,
        num_labels: int = 200,
        multi_label: bool = False,
        input_size: tuple = (3, 244, 244),
        prototypes_per_class: int = 10,
        prototype_channel_num: int = 1280,
        num_prototypes: int = 0,  # important if the network is pruned
    ) -> None:

        super().__init__()
        # region Parameters ------------------------------------------------------------------------
        self.num_labels: int = num_labels
        self.train_mode: str = ""
        self.multi_label: bool = multi_label
        self.multi_label_threshold = None

        self._num_prototypes: int = num_labels * prototypes_per_class

        # If the network is pruned, the number of prototypes can be set to a lower value
        if num_prototypes > 0:
            self._num_prototypes: int = num_prototypes

        self.prototype_shape: tuple[int, int, int, int] = (
            self._num_prototypes,
            prototype_channel_num,
            1,
            1,
        )
        self.active_prototypes: torch.Tensor = torch.ones(self._num_prototypes).cuda()
        # endregion --------------------------------------------------------------------------------

        # region Model Layers ----------------------------------------------------------------------
        self.backbone: nn.Module = backbone

        # the tuple index is currently used if the backbone is an efficientnet with 2 outputs
        # we need the second output, which is the output after pooling
        # backbone_output_size: int | tuple[int, int, int] = model_output_size(
        #     backbone, input_size, tuple_idx=1
        # )
        # assert backbone_output_size == prototype_channel_num

        backbone_out_size = model_output_size(backbone, input_size)

        self.input_size = input_size

        # self.add_on_layers: nn.Module = nn.Sigmoid()

        self.add_on_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=backbone_out_size[0],
                out_channels=self.prototype_shape[1],
                kernel_size=1,
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid(),
        )
        self.prototype_layer = L2SquaredConv2d()
        self.similarity_layer = LogActivation()
        self.last_layer = nn.Linear(self._num_prototypes, self.num_labels, bias=False)
        # endregion --------------------------------------------------------------------------------

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        # use a gaussian distribution for the prototype vectors with mean as the model parameter
        # mean
        # self.prototype_vectors = nn.Parameter(
        #     torch.FloatTensor(*self.prototype_shape).uniform_(
        #         compute_mean_weight(backbone), 0.1
        #     ),
        #     requires_grad=True,
        # )

        # the uniform is used specifically for the efficientnet
        # self.prototype_vectors = nn.Parameter(
        #     torch.FloatTensor(*self.prototype_shape).uniform_(0.4, 1.0),
        #     requires_grad=True,
        # )
        self.prototype_class_identity: torch.Tensor = torch.zeros(
            self._num_prototypes, self.num_labels
        ).cuda()
        for j in range(self._num_prototypes):
            self.prototype_class_identity[j, j // prototypes_per_class] = 1

        self._last_layer_incorrect_connection(0.0)
        self.coefs: dict | None = None
        self.classification_loss_fn: nn.Module | None = None

    def compile(
        self,
        coefs: dict | None = {
            "crs_ent": 1,
            "clst": 0.8,
            "sep": 0.2,
            "div": 0.2,
            "l1": 1e-3,
        },
        class_weights: list | None = None,
    ) -> None:
        self.coefs: dict | None = coefs

        # region Dataset Class Weights -------------------------------------------------------------
        if class_weights is None:
            weights = torch.ones(self.num_labels, dtype=torch.float32)
        else:
            assert len(class_weights) == self.num_labels
            weights = torch.tensor(class_weights, dtype=torch.float32)
        # endregion --------------------------------------------------------------------------------

        # region Loss Function ---------------------------------------------------------------------
        self.classification_loss_fn = (
            nn.CrossEntropyLoss(weights)
            if not self.multi_label
            else nn.MultiLabelMarginLoss(reduction="sum")
        )
        # endregion --------------------------------------------------------------------------------

    def warmup(self) -> None:
        """
        Set the model into warmup mode.
        """
        # freeze all feature layers
        for p in self.backbone.parameters():
            p.requires_grad = False

        # unfreeze add_on_layers
        for p in self.add_on_layers.parameters():
            p.requires_grad = True

        # unfreeze prototype_vectors
        self.prototype_vectors.requires_grad = True

        # freeze last layer
        for p in self.last_layer.parameters():
            p.requires_grad = False

        self.train_mode = "warmup"

    def joint(self) -> None:
        """
        Set the model into joint mode.
        """
        # unfreeze all feature layers
        for p in self.backbone.parameters():
            p.requires_grad = True

        # unfreeze add_on_layers
        for p in self.add_on_layers.parameters():
            p.requires_grad = True

        # unfreeze prototype_vectors
        self.prototype_vectors.requires_grad = True

        # unfreeze last layer
        for p in self.last_layer.parameters():
            p.requires_grad = False

        self.train_mode = "joint"

    def fine_tune(self) -> None:
        """
        Set the model into fine-tune mode.
        """
        # freeze all feature layers
        for p in self.backbone.parameters():
            p.requires_grad = False

        # unfreeze add_on_layers
        for p in self.add_on_layers.parameters():
            p.requires_grad = False

        # unfreeze prototype_vectors
        self.prototype_vectors.requires_grad = False

        # unfreeze last layer
        for p in self.last_layer.parameters():
            p.requires_grad = True

        self.train_mode = "fine_tune"

    def num_classes(self) -> int:
        return self.num_labels

    def num_prototypes(self) -> int:
        return self._num_prototypes

    def _last_layer_incorrect_connection(self, incorrect_strength) -> None:
        """
        the incorrect strength will be actual strength if -0.5 then input -0.5
        """
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength

        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return logits

    def explain(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # save the Batch size
        batch_size = x.size(0)
        mask_size = x.size(1)
        # reshape the input tensor to the correct shape
        x = x.view(-1, *self.input_size)

        # The first step is to calculate the feature maps.
        x = self.backbone(x)

        # check if x is list, this is in the efficientnet case
        # efficientnet specific
        if not hasattr(x, "shape"):
            x = x[1].unsqueeze(2).unsqueeze(3)
        x = self.add_on_layers(x)

        # The second step is to calculate the distance between each prototype
        # and each feature vector. We use the novel L2SquaredConv2d layer for
        # this purpose.
        distance_maps = self.prototype_layer(x, self.prototype_vectors)

        # This is an intermediate step that is only used for evaluation purposes.
        similarity_maps = self.similarity_layer(distance_maps)
        similarity_maps = similarity_maps.view(batch_size, mask_size, -1)

        # the third step is to get the minimum distance from the distance maps
        # print("distance_maps.size()[2:]: ", distance_maps.size()[:])
        min_distances = min_pool2d(distance_maps, kernel_size=distance_maps.size()[2:]).squeeze()

        min_distances = min_distances.view(batch_size, mask_size, -1)
        min_distances = torch.min(min_distances, dim=1)[0]

        # the fourth step is to calculate the similarity scores based on the
        similarity_scores = self.similarity_layer(min_distances)

        # the last step is to calculate the logits based on the similarity
        # scores and the last layer weights
        logits = self.last_layer(similarity_scores)

        # return the logits and the minimum distances because there are needed
        # for the loss calculation
        # return logits, min_distances, similarity_maps
        x = x.view(batch_size, mask_size, -1)

        return (
            logits,
            similarity_maps.permute(0, 2, 1).unsqueeze(-1),
            x.permute(0, 2, 1),
        )

    def get_prototypes(self, idx) -> torch.Tensor:
        """
        Get N prototypes for each batch

        Args:
            idx: the prototype index BxN
        """

        return self.prototype_vectors[idx]

    def global_explanation_size(self, epsilon: float) -> int:
        num_non_zero_prototypes = torch.gt(self.last_layer.weight, epsilon).any(dim=0).sum().item()
        return num_non_zero_prototypes

    def classification_sparsity(self, epsilon):
        num_weights = torch.numel(self.last_layer.weight)

        num_positive_weights = torch.count_nonzero(self.last_layer.weight > epsilon)
        num_negative_weights = torch.count_nonzero(self.last_layer.weight < -epsilon)

        num_non_zero_weights = num_positive_weights + num_negative_weights

        sparsity = (num_weights - num_non_zero_weights) / num_weights

        return sparsity.item()

    def negative_positive_reasoning_ratio(self, epsilon):

        num_positive_weights = torch.count_nonzero(self.last_layer.weight > epsilon)
        num_negative_weights = torch.count_nonzero(self.last_layer.weight < -epsilon)

        return (num_negative_weights / num_positive_weights).item()

    def prune_prototypes(self, prototypes_to_prune):
        """
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        """
        prototypes_to_keep = list(set(range(self._num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(
            self.prototype_vectors.data[prototypes_to_keep, ...], requires_grad=True
        ).cuda()

        self.prototype_shape = list(self.prototype_vectors.size())
        self._num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self._num_prototypes
        self.last_layer.out_features = self.num_labels
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def forward(self, x):
        # save the Batch size
        batch_size = x.size(0)
        mask_size = x.size(1)
        # reshape the input tensor to the correct shape
        x = x.view(-1, *self.input_size)

        # The first step is to calculate the feature maps.
        x = self.backbone(x)

        # efficientnet specific
        if not hasattr(x, "shape"):
            x = x[1].unsqueeze(2).unsqueeze(3)
        x = self.add_on_layers(x)

        # The second step is to calculate the distance between each prototype
        # and each feature vector. We use the novel L2SquaredConv2d layer for
        # this purpose.
        min_distances = self.prototype_layer(x, self.prototype_vectors).squeeze()
        # print("prototype_vectors.shape: ", self.prototype_vectors.shape)
        # print("min_distances.shape: ", min_distances.shape)

        # This is an intermediate step that is only used for evaluation purposes.
        # similarity_maps = self.similarity_layer(distance_maps)
        # similarity_maps = similarity_maps.view(batch_size, mask_size, -1)

        # the third step is to get the minimum distance from the distance maps
        # min_distances = min_pool2d(distance_maps, kernel_size=distance_maps.size()[2:]).squeeze()

        min_distances = min_distances.view(batch_size, mask_size, -1)
        min_distances = torch.min(min_distances, dim=1)[0]

        # the fourth step is to calculate the similarity scores based on the
        similarity_scores = self.similarity_layer(min_distances)

        # the last step is to calculate the logits based on the similarity
        # scores and the last layer weights
        logits = self.last_layer(similarity_scores)
        # print("logits.shape: ", logits.shape)
        # print("similarity_scores.shape: ", similarity_scores.shape)

        # return the logits and the minimum distances because there are needed
        # for the loss calculation
        # return logits, min_distances, similarity_maps
        return logits, min_distances

    def push_forward(self, x):
        # save the Batch size
        batch_size = x.size(0)
        mask_size = x.size(1)
        # reshape the input tensor to the correct shape
        x = x.view(-1, *self.input_size)

        # The first step is to calculate the feature maps.
        x = self.backbone(x)

        # efficientnet specific
        if not hasattr(x, "shape"):
            x = x[1].unsqueeze(2).unsqueeze(3)
        x = self.add_on_layers(x)

        # The second step is to calculate the distance between each prototype
        # and each feature vector. We use the novel L2SquaredConv2d layer for
        # this purpose.
        min_distances = self.prototype_layer(x, self.prototype_vectors).squeeze()

        x = x.view(batch_size, mask_size, -1, 1, 1)

        # This is an intermediate step that is only used for evaluation purposes.
        # similarity_maps = self.similarity_layer(distance_maps)
        # similarity_maps = similarity_maps.view(batch_size, mask_size, -1)

        # the third step is to get the minimum distance from the distance maps
        # min_distances = min_pool2d(distance_maps, kernel_size=distance_maps.size()[2:]).squeeze()

        min_distances = min_distances.view(batch_size, mask_size, -1)

        return x, min_distances


def compute_loss(
    pl_model,
    output,
    target,
) -> tuple[dict[str, float], torch.Tensor]:
    """
    Computes the total loss based on various components and their coefficients.
    Returns:
    Tensor: The computed total loss.
    """
    log = {}

    logits = output[0]
    min_distances = output[1]

    max_dist: float = (
        pl_model.model.prototype_shape[1]
        * pl_model.model.prototype_shape[2]
        * pl_model.model.prototype_shape[3]
    )

    if pl_model.model.prototype_class_identity.device != logits.device:
        prototype_class_identity = pl_model.model.prototype_class_identity.to(logits.device)
    else:
        prototype_class_identity = pl_model.model.prototype_class_identity

    # Calculate cross-entropy loss
    if pl_model.model.multi_label:
        target = convert_to_multilabelmargin_input(target)
        cross_entropy = pl_model.model.classification_loss_fn(logits, target)
    else:
        # Calculate cross-entropy loss
        cross_entropy = pl_model.model.classification_loss_fn(logits, target)

    log["classification_loss"] = cross_entropy.item()

    if pl_model.model.train_mode == "warmup" or pl_model.model.train_mode == "joint":
        if pl_model.model.multi_label:
            target = target.unsqueeze(2)  # bxcx1
            prototypes_of_correct_class = target * prototype_class_identity.transpose(0, 1)
            inverted_distances, _ = torch.max(
                (max_dist - min_distances.unsqueeze(1)) * prototypes_of_correct_class,
                dim=2,
            )

            inverted_distances = (max_dist - inverted_distances) * target.squeeze(2)
            inverted_distances = torch.sum(inverted_distances, dim=1) / torch.sum(
                target.squeeze(2), dim=1
            )
            cluster_cost = torch.mean(inverted_distances)
            log["cluster_cost"] = cluster_cost.item()

            prototypes_of_wrong_class = (1 - target) * prototype_class_identity.transpose(0, 1)

            inverted_distances, _ = torch.max(
                (max_dist - min_distances.unsqueeze(1)) * prototypes_of_wrong_class,
                dim=2,
            )

            inverted_distances = (max_dist - inverted_distances) * -(target.squeeze(2) - 1)
            inverted_distances = torch.sum(inverted_distances, dim=1) / torch.sum(
                -(target.squeeze(2) - 1), dim=1
            )

            separation_cost = torch.mean(inverted_distances)
            log["separation_cost"] = separation_cost.item()

        else:
            prototypes_of_correct_class = torch.t(prototype_class_identity[:, target])
            inverted_distances, _ = torch.max(
                (max_dist - min_distances) * prototypes_of_correct_class, dim=1
            )
            cluster_cost = torch.mean(max_dist - inverted_distances)
            log["cluster_cost"] = cluster_cost.item()

            # Calculate separation cost from wrong class prototypes
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = torch.max(
                (max_dist - min_distances) * prototypes_of_wrong_class, dim=1
            )
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
            log["separation_cost"] = separation_cost.item()

            # div loss
            div_loss = compute_div_loss(pl_model.model, prototype_class_identity, target).item()
            log["div_loss"] = div_loss

    if pl_model.model.train_mode == "fine_tune":
        # Apply L1 regularization
        l1_mask = 1 - torch.t(prototype_class_identity)
        l1 = (pl_model.model.last_layer.weight * l1_mask).norm(p=1)
        log["l1"] = l1.item()

    if pl_model.model.train_mode == "warmup" or pl_model.model.train_mode == "joint":
        total_loss = (
            pl_model.model.coefs["crs_ent"] * cross_entropy
            + pl_model.model.coefs["clst"] * cluster_cost
            - pl_model.model.coefs["sep"] * separation_cost
            + pl_model.model.coefs["div"] * div_loss
        )
    elif pl_model.model.train_mode == "fine_tune":
        total_loss = (
            pl_model.model.coefs["crs_ent"] * cross_entropy + pl_model.model.coefs["l1"] * l1
        )
    else:
        raise ValueError("Invalid train mode")
    log["loss"] = total_loss.item()

    return log, total_loss


def compute_div_loss(base_model, prototype_class_identity, target):
    # print("prototype_class_identity.shape: ", prototype_class_identity.shape)
    # print("target.shape: ", target.shape)
    # target Bx1
    if base_model.multi_label:
        target = target.unsqueeze(2)  # bxcx1
        prototype_ids = target * prototype_class_identity.transpose(0, 1)
        print("The div loss with multi label is not tested")
    else:
        prototype_ids = prototype_class_identity[:, target]  # N x B
    prototype_ids = prototype_ids.permute(1, 0)  # B x N

    # get the indices where the prototype_ids is 1
    prototype_ids = torch.nonzero(prototype_ids, as_tuple=True)[1]

    prototype_vectors = base_model.prototype_vectors[prototype_ids, :, :, :]  # B*N x C x 1 x 1

    # reshape the prototype_vectors to be B x N x C
    # prototype_vectors = prototype_vectors.view(
    #     target.size(0), -1, prototype_vectors.size(1)
    # )
    prototype_vectors = prototype_vectors.reshape(target.size(0), -1, prototype_vectors.size(1))

    # Expand to B x 1 x N x C and B x N x 1 x C
    diff = prototype_vectors.unsqueeze(1) - prototype_vectors.unsqueeze(2)  # B x N x N x C

    distance_matrix = torch.linalg.norm(diff, dim=-1, ord=2)  # B x N x N

    # sum the distance for each B
    distance_matrix = torch.sum(torch.sum(distance_matrix, dim=-1), dim=-1)  # B

    # alpha_div_loss=-0.001
    div_loss = torch.exp(-0.001 * distance_matrix)  # B
    div_loss = torch.mean(div_loss)  # 1

    return div_loss
