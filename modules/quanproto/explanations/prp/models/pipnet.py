"""
This module contains the PIPNetPRP class which is a PIPNet model with PRP
explanation capabilities.
"""

import torch
import torch.nn as nn
import quanproto.explanations.prp.utils.prp as prp
import copy
from quanproto.models.helper import model_output_size

from quanproto.models.pipnet.pipnet import PIPNet
from quanproto.explanations.interfaces.explanation_interface import ExplanationInterface


class PIPNetPRP(PIPNet, ExplanationInterface):
    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        input_size=(3, 224, 224),
        multi_label: bool = False,
    ):
        super(PIPNetPRP, self).__init__(
            num_classes=num_classes,
            backbone=backbone,
            input_size=input_size,
            multi_label=multi_label,
        )

    def canonize(self, input_size: tuple, feature: str = "resnet50") -> None:

        # region Canonized backbone model --------------------------------------------------------------
        canonized_backbone = prp.canonized_feature_dict[feature]()
        canonized_backbone.copy_weights(
            self.backbone, lrp_params=prp.lrp_params_def1, lrp_layer2method=prp.lrp_layer2method
        )
        self.backbone = canonized_backbone

        backbone_output_size = model_output_size(self.backbone, input_size)
        # endregion -----------------------------------------------------------------------------------

        self.softmax = prp.get_lrpwrapperformodule(
            copy.deepcopy(torch.nn.Softmax(dim=1)), prp.lrp_params_def1, prp.lrp_layer2method
        )

        # region Canonized max layer -----------------------------------------------------------------
        self.max_layer = prp.get_lrpwrapperformodule(
            copy.deepcopy(torch.nn.MaxPool2d(backbone_output_size[1:], return_indices=False)),
            prp.lrp_params_def1,
            prp.lrp_layer2method,
        )
        # endregion -----------------------------------------------------------------------------------

    def relevance_map(self, image: torch.Tensor, prototype_idx: int) -> torch.Tensor:
        """
        Compute the relevance map for a single image.

        Args:
            image: The input image (channels, height, width)
            prototype_idx: The prototype index

        Returns:
            The relevance map for the image. (channels, height, width)
        """

        # we can only compute the relevance map for one image at a time because
        # we can only use the backward function for a scalar value, the
        # similarity of one prototype for one image.
        image = image.unsqueeze(0)

        # we need the gradient values for the image because these are the
        # relevance values
        image.requires_grad = True

        with torch.enable_grad():
            feature = self.backbone(image)

            # a 1 will be used as backward value for the max layer so the softmax layer can be ignored
            # similarity_maps = self.softmax(feature)

            # global max pooling so that we get one value per map
            similarity_maps = self.max_layer(feature)
            similarity_maps = similarity_maps.view(-1, similarity_maps.shape[1])

            (similarity_maps[:, prototype_idx]).backward()

        # pixel level relevance
        relevance_map = torch.sum(image.grad.data.squeeze(0), dim=0)
        # only positive relevance
        relevance_map = torch.nn.functional.relu(relevance_map)

        return relevance_map

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
        saliency_maps = torch.stack(
            [torch.stack([self.relevance_map(x[b], i) for i in prototype_ids[b]]) for b in range(B)]
        )

        return saliency_maps
