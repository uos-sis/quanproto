"""
This module contains the ProtoPNetUpscale class, which is a subclass of
ProtoPNet that is used for the Upscale explanation method.

The upscale explanation method is model agnostic so the canonize method is not
implemented.
"""

import torch
import torch.nn as nn
import cv2


from quanproto.models.protopnet.protopnet import ProtoPNet
from quanproto.explanations.interfaces.explanation_interface import ExplanationInterface


class ProtoPNetUpscale(ProtoPNet, ExplanationInterface):
    def __init__(
        self,
        backbone: nn.Module,
        num_labels: int = 200,
        multi_label: bool = False,
        input_size: tuple = (3, 244, 244),
        prototypes_per_class: int = 10,
        prototype_shape: tuple = (128, 1, 1),
        num_prototypes: int = 0,
    ) -> None:
        super(ProtoPNetUpscale, self).__init__(
            backbone=backbone,
            num_labels=num_labels,
            multi_label=multi_label,
            input_size=input_size,
            prototypes_per_class=prototypes_per_class,
            prototype_shape=prototype_shape,
            num_prototypes=num_prototypes,
        )

    def canonize(self) -> None:
        pass

    def saliency_maps(self, x, prototype_ids):
        """
        Compute the saliency maps for the input data for each prototype

        Args:
            x: The input data (B, channels, height, width)
            prototype_ids: The prototype indices (B, num_ids)

        Returns:
            The saliency maps (B, num_prototypes, height, width)
        """

        B = x.size(0)
        assert prototype_ids.size(0) == B

        self.eval()
        input_size = x.size()[2:]
        _, similarity_maps, _ = self.explain(x)

        saliency_maps = torch.stack(
            [
                torch.stack(
                    [
                        torch.from_numpy(
                            cv2.resize(
                                similarity_maps[b][i].detach().cpu().numpy(),
                                dsize=input_size,
                                interpolation=cv2.INTER_CUBIC,
                            )
                        ).cuda()
                        for i in prototype_ids[b]
                    ]
                )
                for b in range(B)
            ]
        )

        return saliency_maps
