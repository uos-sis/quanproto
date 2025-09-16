import copy

import cv2
import torch
import torch.nn as nn

import quanproto.explanations.prp.utils.prp as prp
from quanproto.explanations.interfaces.explanation_interface import ExplanationInterface
from quanproto.metrics import helpers
from quanproto.models.helper import model_output_size
from quanproto.models.protomask.protomask import ProtoMask
from quanproto.utils.vis_helper import save_image_mask, save_mask


class ProtoMaskPRP(ProtoMask, ExplanationInterface):
    def __init__(
        self,
        backbone: nn.Module,
        num_labels: int,
        input_size=(3, 224, 224),
        multi_label: bool = False,
        prototypes_per_class: int = 10,
        prototype_channel_num: int = 1280,
        num_prototypes: int = 0,
    ):
        super(ProtoMaskPRP, self).__init__(
            num_labels=num_labels,
            backbone=backbone,
            input_size=input_size,
            multi_label=multi_label,
            prototypes_per_class=prototypes_per_class,
            prototype_channel_num=prototype_channel_num,
            num_prototypes=num_prototypes,
        )
        self.explanation_type = "prp"

    def canonize(self, input_size: tuple, feature: str = "resnet50") -> None:

        # region Canonized backbone model --------------------------------------------------------------
        canonized_backbone = prp.canonized_feature_dict[feature]()
        canonized_backbone.copy_weights(
            self.backbone,
            lrp_params=prp.lrp_params_def1,
            lrp_layer2method=prp.lrp_layer2method,
        )
        self.backbone = canonized_backbone

        backbone_output_size = model_output_size(self.backbone, input_size)
        # endregion -----------------------------------------------------------------------------------

        canonized_add_on_layer = [
            prp.get_lrpwrapperformodule(
                copy.deepcopy(layer), prp.lrp_params_def1, prp.lrp_layer2method
            )
            for _, layer in self.add_on_layers.named_modules()
            if not isinstance(layer, nn.Sequential)
        ]
        self.add_on_layers = nn.Sequential(*canonized_add_on_layer)

    def relevance_map(self, masks: torch.Tensor, prototype_idx: int) -> torch.Tensor:
        """
        Compute the relevance map for a single image.

        Args:
            masks: The input image (M,channels, height, width)
            prototype_idx: The prototype index

        Returns:
            The relevance map for the image. (channels, height, width)
        """

        # First find the mask with the highest similarity to the prototype
        with torch.no_grad():
            feature = self.backbone(masks)
            feature = self.add_on_layers(feature)
            prototype = self.prototype_vectors[prototype_idx].unsqueeze(0)
            distance_maps = self.prototype_layer(feature, prototype)
            similarity_maps = self.similarity_layer(distance_maps).squeeze()
            mask_idx = torch.max(similarity_maps, dim=0)[1]

        mask = masks[mask_idx].unsqueeze(0)
        mask.requires_grad = True

        with torch.enable_grad():
            feature = self.backbone(mask)
            feature = self.add_on_layers(feature)

            # use the PRP version of the L2Conv2d
            newl2 = prp.L2Conv2dLRPFunction.apply
            prototype = self.prototype_vectors[prototype_idx].unsqueeze(0)
            similarity = newl2(feature, prototype).squeeze()
            similarity.backward()

        # pixel level relevance
        relevance_map = torch.sum(mask.grad.data.squeeze(0), dim=0)
        # only positive relevance
        relevance_map = torch.nn.functional.relu(relevance_map)

        return relevance_map, mask_idx

    def invert_cropping(self, mask, mask_bbox, mask_size):
        """
        Invert the cropping of the mask to get the original image size.

        Args:
            mask: The input mask (channels, height, width)
            mask_bboxes: The bounding boxes for the masks (4)
            mask_size: The size of the masks (2)

        Returns:
            The inverted mask (channels, height, width)
        """

        # 1) compute the original size of the mask with the bounding box
        height = (mask_bbox[3] - mask_bbox[1]).item()
        width = (mask_bbox[2] - mask_bbox[0]).item()
        # print(f"height: {height}, width: {width}")
        # print(f"mask_size: {mask_size}")
        # # 2) make a resize with opencv
        # mask = torch.nn.functional.interpolate(
        #     mask.unsqueeze(0).unsqueeze(0),
        #     size=(int(height), int(width)),
        #     mode="bilinear",
        #     align_corners=False,
        # )

        # 3) compute the scaling ratio of the entire image
        height_scale = (self.input_size[1] / mask_size[0]).item()
        width_scale = (self.input_size[2] / mask_size[1]).item()

        tranformed_height = round(height * height_scale)
        tranformed_width = round(width * width_scale)

        # 4) resize the mask with the original size to the ratio of the transformed image so segmentation mask ans this mask overlap correctly
        mask = (
            torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(tranformed_height, tranformed_width),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        # 5) create a mask with the original size
        new_mask = torch.zeros(
            (self.input_size[1], self.input_size[2]),
            dtype=torch.float32,
            device=mask.device,
        )

        start_width = round(mask_bbox[0].item() * width_scale)
        start_height = round(mask_bbox[1].item() * height_scale)
        end_width = round(mask_bbox[2].item() * width_scale)
        end_height = round(mask_bbox[3].item() * height_scale)

        # 5) adjust rounding errors
        if end_width - start_width != tranformed_width:
            end_width = start_width + tranformed_width
        if end_height - start_height != tranformed_height:
            end_height = start_height + tranformed_height

        assert (
            end_width <= self.input_size[2] or end_height <= self.input_size[1]
        ), f"end_width: {end_width}, end_height: {end_height}, input_size: {self.input_size}"

        # 6) copy the mask to the original size
        new_mask[start_height:end_height, start_width:end_width] = mask

        return new_mask

    def saliency_maps(
        self, x, explanation_masks, mask_bboxes, mask_size, prototype_ids
    ):
        """
        Compute the saliency maps for the input data for each prototype

        Args:
            x: The input data (B, M, channels, height, width)
            explanation_masks: The explanation masks (B, M, channels, height, width)
            prototype_ids: The prototype indices (B, num_ids)
            mask_bboxes: The bounding boxes for the masks (B, M, 4)
            mask_size: The size of the masks (B, 2)

        Returns:
            The saliency maps (B, num_prototypes, height, width)
        """

        B = x.size(0)
        assert prototype_ids.size(0) == B

        self.eval()

        # saliency_maps = torch.stack(
        #     [torch.stack([self.invert_cropping(self.relevance_map(x[b], i), mask for i in prototype_ids[b]]) for b in range(B)]
        # )
        saliency_maps = torch.empty(0, device=x.device)
        mask_ids = []
        for b in range(B):
            prototype_masks = torch.empty(0, device=x.device)
            for i in prototype_ids[b]:
                # print(f"Computing saliency map for image {b} and prototype {i}")
                mask, mask_id = self.relevance_map(x[b], i)
                mask_ids.append(mask_id)

                # invert the cropping of the mask to get the original image size
                # TODO: needs to be commented out for the PRP version
                mask = self.invert_cropping(mask, mask_bboxes[b][mask_id], mask_size[b])

                prototype_masks = torch.cat((prototype_masks, mask.unsqueeze(0)), dim=0)
            saliency_maps = torch.cat(
                (saliency_maps, prototype_masks.unsqueeze(0)), dim=0
            )

        return saliency_maps
        # TODO:  needs to be commented out for the PRP version
        return saliency_maps, mask_ids
