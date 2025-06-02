"""
This module contains the PIPNetUpscale class, which is a subclass of PIPNet that
is used for the Upscale explanation method.

The upscale explanation method is model agnostic so the canonize method is not
implemented.
"""

import torch
import torch.nn as nn
import cv2

from quanproto.models.pipnet.pipnet import PIPNet
from quanproto.explanations.interfaces.explanation_interface import ExplanationInterface


class PIPNetUpscale(PIPNet, ExplanationInterface):
    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        input_size=(3, 224, 224),
        multi_label: bool = False,
    ):
        super(PIPNetUpscale, self).__init__(
            num_classes=num_classes,
            backbone=backbone,
            input_size=input_size,
            multi_label=multi_label,
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
