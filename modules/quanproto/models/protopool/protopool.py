"""
This file contains the ProtoPool model.
The implementation is based on the original ProtoPool repository

Reference: https://github.com/gmum/ProtoPool

"""

from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax

from quanproto.functional.activation import LogActivation
from quanproto.functional.l2conv import L2SquaredConv2d
from quanproto.functional.minpool import min_pool2d
from quanproto.models.helper import convert_to_multilabelmargin_input, model_output_size
from quanproto.models.interfaces.prototype_model_interfaces import (
    PrototypeModelInterface,
)


class GumbleScaleCallback(L.Callback):
    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        pl_module.model.gumble_scalar(pl_module.current_epoch)


class ProtoPool(nn.Module, PrototypeModelInterface):

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 200,
        multi_label: bool = False,
        input_size: tuple = (3, 244, 244),
        prototype_shape: tuple = (128, 1, 1),
        num_prototypes: int = 200,
        num_descriptive: int = 10,
    ) -> None:
        super().__init__()

        # region Parameters ------------------------------------------------------------------------
        self._num_classes = num_classes
        self.multi_label = multi_label
        self.multi_label_threshold = None
        self.scalar = 0.0

        self.num_descriptive = num_descriptive
        self._num_prototypes = num_prototypes
        self.prototype_shape: tuple[int, int, int, int] = (
            self._num_prototypes,
        ) + prototype_shape
        # endregion --------------------------------------------------------------------------------

        # region Model Layers ----------------------------------------------------------------------
        self.backbone = backbone
        backbone_output_size = model_output_size(backbone, input_size)

        self.add_on_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=backbone_output_size[0],
                out_channels=self.prototype_shape[1],
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.prototype_layer = L2SquaredConv2d()
        self.similarity_layer = LogActivation()

        self.last_layer = nn.Linear(
            self.num_descriptive * self._num_classes, self._num_classes, bias=False
        )
        # endregion --------------------------------------------------------------------------------

        self.proto_presence = nn.Parameter(
            torch.zeros(self._num_classes, self._num_prototypes, num_descriptive),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.proto_presence, gain=1.0)

        self.prototype_vectors = nn.Parameter(
            torch.rand(self.prototype_shape), requires_grad=True
        )

        self.prototype_class_identity = torch.zeros(
            self.num_descriptive * self._num_classes, self._num_classes
        ).cuda()
        for j in range(self.num_descriptive * self._num_classes):
            self.prototype_class_identity[j, j // self.num_descriptive] = 1
        self._initialize_weights()

    def compile(
        self,
        coefs: dict | None = {
            "crs_ent": 1,
            "clst": 0.8,
            "sep": 0.08,
            "orth": 1.0,
            "l1": 1e-3,
            "tau_start": 1.0,
            "tau_end": 1e-3,
            "decreasing_interval": 30,
        },
        class_weights: list | None = None,
    ) -> None:
        self.coefs = coefs
        # region Dataset Class Weights -------------------------------------------------------------
        if class_weights is None:
            weights = torch.ones(self._num_classes, dtype=torch.float32)
        else:
            assert len(class_weights) == self._num_classes
            weights = torch.tensor(class_weights, dtype=torch.float32)
        # endregion --------------------------------------------------------------------------------

        # region Loss Function ---------------------------------------------------------------------
        self.classification_loss_fn = (
            nn.CrossEntropyLoss(weights)
            if not self.multi_label
            else nn.MultiLabelMarginLoss()
        )
        # endregion --------------------------------------------------------------------------------

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

    def _initialize_weights(self) -> None:
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self._last_layer_incorrect_connection(0.0)

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
        self.proto_presence.requires_grad = True

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
        self.proto_presence.requires_grad = True

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
        self.proto_presence.requires_grad = False

        # unfreeze last layer
        for p in self.last_layer.parameters():
            p.requires_grad = True

        self.train_mode = "fine_tune"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        if self.scalar == 0:
            proto_presence = torch.softmax(self.proto_presence, dim=1)
        else:
            # proto_presence = gumbel_softmax(self.proto_presence * gumbel_scale, dim=1, tau=0.5)
            proto_presence = gumbel_softmax(
                self.proto_presence * self.scalar, dim=1, tau=1.0
            )

        x = self.backbone(x)

        # efficientnet specific
        if not hasattr(x, "shape"):
            x = x[0]

        x = self.add_on_layers(x)

        distance_maps = self.prototype_layer(x, self.prototype_vectors)

        # intermediate step for evaluation
        # similarity_maps = self.similarity_layer(distance_maps)

        min_distances = min_pool2d(distance_maps, kernel_size=distance_maps.size()[2:])
        min_distances = min_distances.view(-1, self._num_prototypes)

        avg_distances = F.avg_pool2d(
            distance_maps, kernel_size=distance_maps.size()[2:]
        )
        avg_distances = avg_distances.view(-1, self._num_prototypes)

        max_similarity = self.similarity_layer(min_distances)
        avg_similarity = self.similarity_layer(avg_distances)
        similarity = max_similarity - avg_similarity

        mixed_similarity = torch.einsum(
            "bp,cpn->bcn", similarity, proto_presence
        )  # [b, c, n]

        logits = self.last_layer(mixed_similarity.flatten(start_dim=1))

        return logits, min_distances, proto_presence  # [b,c,n] [b, p] [c, p, n]

    def push_forward(self, x):
        x = self.backbone(x)
        x = self.add_on_layers(x)

        distance_maps = self.prototype_layer(x, self.prototype_vectors)
        return x, distance_maps

    def explain(self, x):
        # TODO: 0.001 is the tau_end value. We used it here because of lower performance otherwise
        proto_presence = gumbel_softmax(self.proto_presence * 1 / 0.001, dim=1, tau=1.0)

        x = self.backbone(x)

        # efficientnet specific
        if not hasattr(x, "shape"):
            x = x[0]

        x = self.add_on_layers(x)

        distance_maps = self.prototype_layer(x, self.prototype_vectors)

        # intermediate step for evaluation
        similarity_maps = self.similarity_layer(distance_maps)

        min_distances = min_pool2d(distance_maps, kernel_size=distance_maps.size()[2:])
        min_distances = min_distances.view(-1, self._num_prototypes)

        avg_distances = F.avg_pool2d(
            distance_maps, kernel_size=distance_maps.size()[2:]
        )
        avg_distances = avg_distances.view(-1, self._num_prototypes)

        max_similarity = self.similarity_layer(min_distances)
        avg_similarity = self.similarity_layer(avg_distances)
        similarity = max_similarity - avg_similarity

        mixed_similarity = torch.einsum(
            "bp,cpn->bcn", similarity, proto_presence
        )  # [b, c, n]

        logits = self.last_layer(mixed_similarity.flatten(start_dim=1))

        return logits, similarity_maps, x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.explain(x)
        return logits

    def get_prototypes(self, indices: torch.Tensor) -> torch.Tensor:
        return self.prototype_vectors[indices]

    def num_classes(self) -> int:
        return self._num_classes

    def num_prototypes(self) -> int:
        return self._num_prototypes

    def global_explanation_size(self, epsilon):
        # TODO: we have to add the tau_end value here, because the network only uses the scaled
        # version of the prototype_presence matrix in the forward pass
        num_non_zero_prototypes = torch.gt(self.proto_presence * 1 / 0.001, epsilon)
        num_non_zero_prototypes = torch.sum(num_non_zero_prototypes, dim=2)
        num_non_zero_prototypes = torch.sum(num_non_zero_prototypes, dim=0)
        num_non_zero_prototypes = torch.sum(num_non_zero_prototypes > 0).item()

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

    def get_map_class_to_prototypes(self):
        # pp = gumbel_softmax(self.proto_presence / self.coefs["tau_end"], dim=1, tau=0.5).detach()
        pp = gumbel_softmax(
            self.proto_presence / self.coefs["tau_end"], dim=1, tau=1.0
        ).detach()
        return np.argmax(pp.cpu().numpy(), axis=1)

    def gumble_scalar(self, epoch):
        start_val = 1 / self.coefs["tau_start"]
        end_val = 1 / self.coefs["tau_end"]
        epoch_interval = self.coefs["decreasing_interval"]
        alpha = (end_val / start_val) ** 2 / epoch_interval
        scalar = start_val * np.sqrt(alpha * (epoch))

        scalar = max(scalar, start_val)
        scalar = min(scalar, end_val)

        self.scalar = scalar


def dist_loss(base_model, min_distances, proto_presence, top_k, sep=False):
    #         model, [b, p],        [b, p, n],      [scalar]
    max_dist = (
        base_model.prototype_shape[1]
        * base_model.prototype_shape[2]
        * base_model.prototype_shape[3]
    )
    # sum over the descriptive slots of each prototype, so we get a total score that this prototype belongs to this class
    basic_proto = proto_presence.sum(dim=-1).detach()  # [b, p]

    # get the top k prototypes that belong to the class, top_k is the number of descriptive slots
    _, idx = torch.topk(basic_proto, top_k, dim=1)  # [b, n]
    binarized_top_k = torch.zeros_like(basic_proto)
    binarized_top_k.scatter_(1, src=torch.ones_like(basic_proto), index=idx)  # [b, p]
    inverted_distances, _ = torch.max(
        (max_dist - min_distances) * binarized_top_k, dim=1
    )  # [b]
    cost = torch.mean(max_dist - inverted_distances)
    return cost


def compute_loss(
    pl_model,
    output,
    target,
) -> tuple[dict[str, float], torch.Tensor]:
    log = {}

    logits, min_distances, proto_presence = output

    # make a tensor with the same size as the gpu_label tensor but with the arange values
    if pl_model.model.multi_label:
        target = convert_to_multilabelmargin_input(target)
        entropy_loss = pl_model.model.classification_loss_fn(logits, target)
    else:
        # Calculate cross-entropy loss
        entropy_loss = pl_model.model.classification_loss_fn(logits, target)

    log["classification_loss"] = entropy_loss.item()

    orthogonal_loss = torch.Tensor([0]).cuda()
    # Compute cosine similarity between adjacent pairs of descriptive vectors
    for c in range(0, pl_model.model.proto_presence.shape[0]):  # num classes
        cos_sim = torch.nn.functional.cosine_similarity(
            pl_model.model.proto_presence[c, :, :-1].squeeze(-1),
            pl_model.model.proto_presence[c, :, 1:].squeeze(-1),
            dim=0,
        ).sum()
        orthogonal_loss += cos_sim
    orthogonal_loss = orthogonal_loss / (
        pl_model.model.num_descriptive * pl_model.model._num_classes
    )
    log["orthogonal_loss"] = orthogonal_loss.item()

    max_dist = (
        pl_model.model.prototype_shape[1]
        * pl_model.model.prototype_shape[2]
        * pl_model.model.prototype_shape[3]
    )

    if pl_model.model.multi_label:

        total_prototype_presence_per_class = torch.sum(
            proto_presence, dim=2
        )  # [c, p, d]
        # compute the top k prototypes that belong each class
        idx = torch.topk(
            total_prototype_presence_per_class, pl_model.model.num_descriptive, dim=1
        )[1]
        # create a binary mask for the top k prototypes
        top_k_prototypes_per_class = torch.zeros_like(
            total_prototype_presence_per_class
        )
        top_k_prototypes_per_class.scatter_(
            1, src=torch.ones_like(total_prototype_presence_per_class), index=idx
        )

        # OK
        inverted_distances, _ = torch.max(
            (max_dist - min_distances.unsqueeze(1)) * top_k_prototypes_per_class, dim=2
        )

        inverted_distances = (max_dist - inverted_distances) * target
        inverted_distances = torch.sum(inverted_distances, dim=1) / torch.sum(
            target, dim=1
        )
        cluster_cost = torch.mean(inverted_distances)
        log["cluster_cost"] = cluster_cost.item()

        total_prototype_presence_per_class = torch.sum(
            1 - proto_presence, dim=2
        )  # [c, p, d]
        # compute the top k prototypes that belong each class
        idx = torch.topk(
            total_prototype_presence_per_class,
            pl_model.model._num_prototypes - pl_model.model.num_descriptive,
            dim=1,
        )[1]
        # create a binary mask for the top k prototypes
        top_k_prototypes_per_class = torch.zeros_like(
            total_prototype_presence_per_class
        )
        top_k_prototypes_per_class.scatter_(
            1, src=torch.ones_like(total_prototype_presence_per_class), index=idx
        )

        inverted_distances, _ = torch.max(
            (max_dist - min_distances.unsqueeze(1)) * top_k_prototypes_per_class, dim=2
        )

        inverted_distances = (max_dist - inverted_distances) * -(target - 1)
        inverted_distances = torch.sum(inverted_distances, dim=1) / torch.sum(
            -(target - 1), dim=1
        )
        seperation_cost = torch.mean(inverted_distances)
        log["seperation_cost"] = seperation_cost.item()

    else:
        label_p = target.cpu().numpy().tolist()
        proto_presence = proto_presence[label_p]
        inverted_proto_presence = 1 - proto_presence

        cluster_cost = dist_loss(
            pl_model.model,
            min_distances,
            proto_presence,
            pl_model.model.num_descriptive,
        )
        log["cluster_cost"] = cluster_cost.item()

        seperation_cost = dist_loss(
            pl_model.model,
            min_distances,
            inverted_proto_presence,
            pl_model.model._num_prototypes - pl_model.model.num_descriptive,
        )
        log["separation_cost"] = seperation_cost.item()

    l1_mask = 1 - torch.t(pl_model.model.prototype_class_identity)
    l1 = (pl_model.model.last_layer.weight * l1_mask).norm(p=1)
    log["l1"] = l1.item()

    if pl_model.model.train_mode == "warmup" or pl_model.model.train_mode == "joint":
        total_loss = (
            pl_model.model.coefs["crs_ent"] * entropy_loss
            + pl_model.model.coefs["clst"] * cluster_cost
            - pl_model.model.coefs["sep"] * seperation_cost
            + pl_model.model.coefs["orth"] * orthogonal_loss
        )
    elif pl_model.model.train_mode == "fine_tune":
        total_loss = (
            pl_model.model.coefs["crs_ent"] * entropy_loss
            + pl_model.model.coefs["l1"] * l1
        )
    else:
        raise ValueError("Invalid train mode")
    log["loss"] = total_loss.item()

    return log, total_loss
