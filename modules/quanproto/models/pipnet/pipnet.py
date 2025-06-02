"""
This file contains the PIPNet model.
The implementation is based on the original PIPNet repository

reference: https://github.com/M-Nauta/PIPNet
"""

from typing import Any, Mapping

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from quanproto.functional.nonneglinear import NonNegLinear
from quanproto.models.helper import convert_to_multilabelmargin_input, model_output_size
from quanproto.models.interfaces.prototype_model_interfaces import (
    PrototypeModelInterface,
)


class LastLayerCallback(L.Callback):
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch,
        batch_idx,
    ) -> None:
        if pl_module.model.train_mode != "warmup":
            pl_module.eval()
            with torch.no_grad():
                pl_module.model.last_layer.weight.copy_(
                    torch.clamp(pl_module.model.last_layer.weight.data - 1e-3, min=0.0)
                )
                pl_module.model.last_layer.normalization_multiplier.copy_(
                    torch.clamp(
                        pl_module.model.last_layer.normalization_multiplier.data,
                        min=1.0,
                    )
                )
                if pl_module.model.last_layer.bias is not None:
                    pl_module.model.last_layer.bias.copy_(
                        torch.clamp(pl_module.model.last_layer.bias.data, min=0.0)
                    )
            pl_module.train()


class PostProcessCallback(L.Callback):
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        pass


class PIPNet(nn.Module, PrototypeModelInterface):
    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        input_size=(3, 224, 224),
        multi_label: bool = False,
    ):
        super().__init__()

        self._num_classes = num_classes

        # this is used for our awa2 dataset
        self.multi_label = multi_label
        self.multi_label_threshold = None

        self.backbone = backbone
        backbone_output_size = model_output_size(backbone, input_size)

        self._num_prototypes = backbone_output_size[0]

        self.add_on_layers = nn.Sequential(
            nn.Softmax(
                dim=1
            ),  # softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1
        )
        self.pool_layer = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),  # outputs (bs, ps,1,1)
            nn.Flatten(),  # outputs (bs, ps)
        )
        self.last_layer = NonNegLinear(
            self._num_prototypes, self._num_classes, bias=False
        )
        self._init_weights()

    def compile(
        self,
        coefs: dict = {
            "align": 5.0,
            "tanh": 2.0,
            "uniform": 0.0,
            "classification": 2.0,
        },
        class_weights: list = None,
    ) -> None:
        self.coefs = coefs

        if class_weights is None:
            weights = torch.ones(self._num_classes, dtype=torch.float32)
        else:
            assert len(class_weights) == self._num_classes
            weights = torch.tensor(class_weights, dtype=torch.float32)

        self.classification_loss_fn = (
            nn.CrossEntropyLoss(weights)
            if not self.multi_label
            else nn.MultiLabelMarginLoss()
        )

        # get the last block name of the backbone
        last_block_name = list(self.backbone.named_parameters())[-1][0].split(".")[:2]
        last_block_name = ".".join(last_block_name)

        self._last_backbone_block_params = []
        self._backbone_params = []
        for n, p in self.backbone.named_parameters():
            if last_block_name in n:
                self._last_backbone_block_params.append(p)
            else:
                self._backbone_params.append(p)

    def warmup(self) -> None:
        """
        Set the model into warmup mode.
        """
        # freeze backbone
        for p in self._backbone_params:
            p.requires_grad = True

        # unfreeze last conv layer of the feature net
        for p in self._last_backbone_block_params:
            p.requires_grad = True

        for p in self.add_on_layers.parameters():
            p.requires_grad = True

        for p in self.pool_layer.parameters():
            p.requires_grad = True

        self.last_layer.requires_grad = False

        self.train_mode = "warmup"

    def joint(self) -> None:
        """
        Set the model into joint mode.
        """
        # freeze backbone
        for p in self._backbone_params:
            p.requires_grad = True

        # unfreeze last conv layer of the feature net
        for p in self._last_backbone_block_params:
            p.requires_grad = True

        for p in self.add_on_layers.parameters():
            p.requires_grad = True

        for p in self.pool_layer.parameters():
            p.requires_grad = True

        # unfreeze last layer
        # for n, p in self.last_layer.named_parameters():
        #     if "multiplier" in n:
        #         p.requires_grad = False
        #     else:
        #         p.requires_grad = True
        self.last_layer.requires_grad = True

        self.train_mode = "joint"

    def fine_tune(self) -> None:
        """
        Set the model into fine-tune mode.
        """
        # freeze backbone
        for p in self._backbone_params:
            p.requires_grad = False

        # freeze last conv layer of the feature net
        for p in self._last_backbone_block_params:
            p.requires_grad = False

        for p in self.add_on_layers.parameters():
            p.requires_grad = False

        for p in self.pool_layer.parameters():
            p.requires_grad = False

        # unfreeze last layer
        # for n, p in self.last_layer.named_parameters():
        #     if "multiplier" in n:
        #         p.requires_grad = False
        #     else:
        #         p.requires_grad = True
        self.last_layer.requires_grad = True

        self.train_mode = "fine_tune"

    def _init_weights(self) -> None:

        def init_weights_xavier(m):
            if type(m) == torch.nn.Conv2d:
                torch.nn.init.xavier_uniform_(
                    m.weight, gain=torch.nn.init.calculate_gain("sigmoid")
                )

        self.add_on_layers.apply(init_weights_xavier)

        torch.nn.init.normal_(self.last_layer.weight, mean=1.0, std=0.1)
        torch.nn.init.constant_(self.last_layer.normalization_multiplier, val=2.0)

    def forward(self, xs):
        features = self.backbone(xs)

        # efficientnet specific
        if not hasattr(features, "shape"):
            features = features[0]

        features = self.add_on_layers(features)
        similarity_scores = self.pool_layer(features)

        logits = self.last_layer(similarity_scores)  # shape (bs*2, num_classes)

        return logits, similarity_scores, features

    def explain(self, xs):
        features = self.backbone(xs)

        # efficientnet specific
        if not hasattr(features, "shape"):
            features = features[0]

        similarity_maps = self.add_on_layers(features)
        similarity_scores = self.pool_layer(similarity_maps)

        # INFO: The clamped similarity scores do not work in our awa2 experiments
        # clamped_similarity_scores = torch.where(
        #     similarity_scores < 0.1, 0.0, similarity_scores
        # )  # during inference, ignore all prototypes that have 0.1 similarity or lower

        logits = self.last_layer(similarity_scores)  # shape (bs*2, num_classes)

        return logits, similarity_maps, similarity_maps  # is also used as features

    def predict(self, xs):
        logits, _, _ = self.forward(xs)
        return logits

    def get_prototypes(self, indices):
        return None

    def num_classes(self):
        return self._num_classes

    def num_prototypes(self):
        return self._num_prototypes

    def global_explanation_size(self, epsilon):
        num_non_zero_prototypes = (
            torch.gt(self.last_layer.weight, epsilon).any(dim=0).sum().item()
        )
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


def compute_loss(pl_model, output, target):
    log = {}
    EPS = 1e-8
    out, pooled, proto_features = output
    out = torch.log1p(out**pl_model.model.last_layer.normalization_multiplier)

    if pl_model.model.multi_label:
        target = convert_to_multilabelmargin_input(target)

    epoch_frac = (pl_model.current_epoch + 1) / pl_model.warm_epochs

    if pl_model.model.train_mode == "warmup":
        align_pf_weight = epoch_frac
        t_weight = 5.0
        unif_weight = pl_model.model.coefs["uniform"]
        cl_weight = 0.0
    else:
        align_pf_weight = pl_model.model.coefs["align"]
        t_weight = pl_model.model.coefs["tanh"]
        unif_weight = pl_model.model.coefs["uniform"]
        cl_weight = pl_model.model.coefs["classification"]

    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)

    if pf1.shape[0] != pf2.shape[0]:
        # we have an odd batch size delete the last element
        if pf1.shape[0] > pf2.shape[0]:
            pf1 = pf1[:-1]
        else:
            pf2 = pf2[:-1]

    embv2 = pf2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
    embv1 = pf1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)

    a_loss_pf = (
        align_loss(embv1, embv2.detach()) + align_loss(embv2, embv1.detach())
    ) / 2.0
    log["align_loss"] = a_loss_pf.item()

    tanh_loss = (
        -(
            torch.log(torch.tanh(torch.sum(pooled1, dim=0)) + EPS).mean()
            + torch.log(torch.tanh(torch.sum(pooled2, dim=0)) + EPS).mean()
        )
        / 2.0
    )
    log["tanh_loss"] = tanh_loss.item()

    if unif_weight > 0:
        uni_loss = (
            uniform_loss(F.normalize(pooled1 + EPS, dim=1))
            + uniform_loss(F.normalize(pooled2 + EPS, dim=1))
        ) / 2.0
        log["uniform_loss"] = uni_loss.item()

    if not pl_model.model.train_mode == "fine_tune":
        # warmup and joint mode use the align and tanh loss
        loss = align_pf_weight * a_loss_pf
        loss += t_weight * tanh_loss
        # add uniformity loss
        if unif_weight > 0:
            loss += unif_weight * uni_loss

    if not pl_model.model.train_mode == "warmup":
        # joint and fine tune mode use the class loss
        class_loss = pl_model.model.classification_loss_fn(out, target)
        log["classification_loss"] = class_loss.item()

        if pl_model.model.train_mode == "fine_tune":
            # fine tune mode uses only the class loss
            loss = cl_weight * class_loss
        else:
            # joint mode adds the class loss to the previous loss
            loss += cl_weight * class_loss

    # Our tanh-loss optimizes for uniformity and was sufficient for our experiments. However, if pretraining of the prototypes is not working well for your dataset, you may try to add another uniformity loss from https://www.tongzhouwang.info/hypersphere/ Just uncomment the following three lines
    # else:
    #     uni_loss = (
    #         uniform_loss(F.normalize(pooled1 + EPS, dim=1))
    #         + uniform_loss(F.normalize(pooled2 + EPS, dim=1))
    #     ) / 2.0
    #     log["uniform_loss"] = uni_loss.item()
    #     loss += unif_weight * uni_loss

    log["loss"] = loss.item()

    return log, loss


# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. Currently not used but you could try adding it if you want.
def uniform_loss(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss


# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False

    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss
