import torch
import torch.nn as nn

from quanproto.explanations.interfaces.explanation_interface import ExplanationInterface
from quanproto.functional.minpool import min_pool2d
from quanproto.models.protomask.protomask import ProtoMask


class ProtoMaskMask(ProtoMask, ExplanationInterface):
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
        super().__init__(
            backbone=backbone,
            num_labels=num_labels,
            multi_label=multi_label,
            input_size=input_size,
            prototypes_per_class=prototypes_per_class,
            prototype_channel_num=prototype_channel_num,
            num_prototypes=num_prototypes,
        )
        self.explanation_type = "mask"

    def canonize(self) -> None:
        pass

    def saliency_maps(self, x, explanation_masks, prototype_ids):
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

        min_distances = min_distances.view(batch_size, mask_size, -1)
        similarities = self.similarity_layer(min_distances)

        # the similarities have a shape of (batch_size, mask_size, num_prototypes)
        # get the mask id with the highest similarity for each prototype over the batch
        prototype_mask_ids = similarities.argmax(dim=1)  # B x P

        # use only the prototype_ids that are in the prototype_ids list B x K
        row_indices = (
            torch.arange(batch_size).unsqueeze(1).expand(batch_size, prototype_ids.size(1))
        )
        prototype_mask_ids = prototype_mask_ids[row_indices, prototype_ids]

        # get the prototype mask for each prototype
        prototype_masks = explanation_masks[row_indices, prototype_mask_ids]
        # sum the B x K x 3 x H x W to b x K x H x W
        prototype_masks = (prototype_masks.sum(dim=2) > 0).float()

        return prototype_masks
