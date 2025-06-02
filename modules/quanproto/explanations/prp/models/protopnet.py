"""
This file contains the ProtoPNetPRP class, which is used as a wrapper for the
ProtoPNet model.  The ProtoPNetPRP class inherits from the ProtoPNet class and
implements the ExplanationInterface class.  This class is used to compute the
saliency maps for the input data using the PRP method.
"""

import torch
import torch.nn as nn
import quanproto.explanations.prp.utils.prp as prp
import copy


from quanproto.models.protopnet.protopnet import ProtoPNet
from quanproto.explanations.interfaces.explanation_interface import ExplanationInterface
from quanproto.models.helper import model_output_size


class ProtoPNetPRP(ProtoPNet, ExplanationInterface):
    """
    ProtoPNet model with PRP explanation method.
    """

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
        """
        Initializes the ProtoPNetPRP model.

        Args:
            backbone (nn.Module): The backbone network.
            num_labels (int, optional): The number of labels. Defaults to 200.
            multi_label (bool, optional): Whether the model supports multi-label classification. Defaults to False.
            input_size (tuple, optional): The input size of the model. Defaults to (3, 244, 244).
            prototypes_per_class (int, optional): The number of prototypes per class. Defaults to 10.
            prototype_shape (tuple, optional): The shape of each prototype. Defaults to (128, 1, 1).
        """
        super(ProtoPNetPRP, self).__init__(
            backbone=backbone,
            num_labels=num_labels,
            multi_label=multi_label,
            input_size=input_size,
            prototypes_per_class=prototypes_per_class,
            prototype_shape=prototype_shape,
            num_prototypes=num_prototypes,
        )

    def canonize(self, input_size: tuple, feature: str = "resnet50") -> None:
        """
        Canonizes the backbone model, add-on layers, and add a maxpool layer to
        extract the highest value of the similarity maps.

        Args:
            input_size (tuple): The input size of the model. (channels, height, width)
            feature (str, optional): The feature to use for the backbone model.
            Defaults to "resnet50".
        """
        # region Canonized backbone model --------------------------------------
        canonized_backbone = prp.canonized_feature_dict[feature]()
        canonized_backbone.copy_weights(
            self.backbone, lrp_params=prp.lrp_params_def1, lrp_layer2method=prp.lrp_layer2method
        )
        self.backbone = canonized_backbone

        backbone_output_size = model_output_size(self.backbone, input_size)
        # endregion ------------------------------------------------------------

        # region Canonized add on layers ---------------------------------------
        canonized_add_on_layer = [
            prp.get_lrpwrapperformodule(
                copy.deepcopy(layer), prp.lrp_params_def1, prp.lrp_layer2method
            )
            for _, layer in self.add_on_layers.named_modules()
            if not isinstance(layer, nn.Sequential)
        ]
        self.add_on_layers = nn.Sequential(*canonized_add_on_layer)
        # endregion ------------------------------------------------------------

        # region Canonized max layer -------------------------------------------
        self.max_layer = prp.get_lrpwrapperformodule(
            torch.nn.MaxPool2d(backbone_output_size[1:], return_indices=False),
            prp.lrp_params_def1,
            prp.lrp_layer2method,
        )
        # endregion ------------------------------------------------------------

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
            feature = self.add_on_layers(feature)

            # use the PRP version of the L2Conv2d
            newl2 = prp.L2Conv2dLRPFunction.apply
            similarity_maps = newl2(feature, self.prototype_vectors)

            # global max pooling so that we get one value per map
            similarity_maps = self.max_layer(similarity_maps)
            similarity_maps = similarity_maps.view(-1, self.prototype_shape[0])

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
