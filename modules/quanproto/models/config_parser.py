import torch

from quanproto.features.config_parser import feature_dict
from quanproto.models.pipnet import pipnet
from quanproto.models.protomask import protomask
from quanproto.models.protopnet import protopnet
from quanproto.models.protopool import protopool


def create_protomask(
    feature: str = "efficientnet-b0",
    pretrained: bool = True,
    num_classes: int = 200,
    multi_label: bool = False,
    prototypes_per_class: int = 5,
    prototype_channel_num: int = 1280,
    num_prototypes: int = 0,
    model_class=protomask.ProtoMask,
):
    backbone = feature_dict[feature]["model_fn"](pretrained=pretrained)
    input_size = feature_dict[feature]["input_size"]

    model = model_class(
        backbone=backbone,
        num_labels=num_classes,
        multi_label=multi_label,
        input_size=input_size,
        prototypes_per_class=prototypes_per_class,
        num_prototypes=num_prototypes,
        prototype_channel_num=prototype_channel_num,
    )

    return model


def load_protomask(config, state_dict, model_class):
    num_prototypes = 0
    # If the network is pruned, the number of prototypes is saved in the config
    if "num_prototypes" in config:
        num_prototypes = config["num_prototypes"]

    model = create_protomask(
        feature=config["features"],
        pretrained=config["init_feature_weights"],
        num_classes=config["num_classes"],
        multi_label=config["multi_label"],
        prototypes_per_class=config["prototypes_per_class"],
        prototype_channel_num=config["prototype_channel_num"],
        num_prototypes=num_prototypes,
        model_class=model_class,
    )

    state_dict = torch.load(state_dict)
    # check if classification_loss_fn.weight is in the state_dict and remove it
    if "classification_loss_fn.weight" in state_dict:
        del state_dict["classification_loss_fn.weight"]
    model.load_state_dict(state_dict, strict=True)

    return model


def create_protopnet(
    feature: str = "resnet50",
    pretrained: bool = True,
    num_classes: int = 200,
    multi_label: bool = False,
    prototypes_per_class: int = 5,
    prototype_shape: tuple = (128, 1, 1),
    num_prototypes: int = 0,
    model_class=protopnet.ProtoPNet,
):
    backbone = feature_dict[feature]["model_fn"](pretrained=pretrained)
    input_size = feature_dict[feature]["input_size"]

    model = model_class(
        backbone=backbone,
        num_labels=num_classes,
        multi_label=multi_label,
        input_size=input_size,
        prototypes_per_class=prototypes_per_class,
        num_prototypes=num_prototypes,
        prototype_shape=prototype_shape,
    )

    return model


def load_protopnet(config, state_dict, model_class):
    num_prototypes = 0
    if "num_prototypes" in config:
        num_prototypes = config["num_prototypes"]

    model = create_protopnet(
        feature=config["features"],
        pretrained=config["init_feature_weights"],
        num_classes=config["num_classes"],
        multi_label=config["multi_label"],
        prototypes_per_class=config["prototypes_per_class"],
        prototype_shape=tuple(config["prototype_shape"]),
        num_prototypes=num_prototypes,
        model_class=model_class,
    )

    state_dict = torch.load(state_dict)
    model.load_state_dict(state_dict, strict=False)

    return model


def create_protopool(
    feature: str = "resnet50",
    pretrained: bool = True,
    num_classes: int = 200,
    multi_label: bool = False,
    num_prototypes: int = 200,
    num_descriptive: int = 10,
    prototype_shape: tuple = (128, 1, 1),
    model_class=protopool.ProtoPool,
):
    backbone = feature_dict[feature]["model_fn"](pretrained=pretrained)
    input_size = feature_dict[feature]["input_size"]

    model = model_class(
        backbone=backbone,
        num_classes=num_classes,
        multi_label=multi_label,
        input_size=input_size,
        prototype_shape=prototype_shape,
        num_prototypes=num_prototypes,
        num_descriptive=num_descriptive,
    )

    return model


def load_protopool(config, state_dict, model_class):
    model = create_protopool(
        feature=config["features"],
        pretrained=config["init_feature_weights"],
        num_classes=config["num_classes"],
        multi_label=config["multi_label"],
        num_prototypes=config["num_prototypes"],
        num_descriptive=config["descriptives"],
        prototype_shape=tuple(config["prototype_shape"]),
        model_class=model_class,
    )

    state_dict = torch.load(state_dict)
    model.load_state_dict(state_dict, strict=False)

    return model


def create_pipnet(
    feature: str = "resnet50",
    pretrained: bool = True,
    num_classes: int = 200,
    multi_label: bool = False,
    model_class=pipnet.PIPNet,
):
    backbone = feature_dict[feature]["model_fn"](pretrained=pretrained)
    input_size = feature_dict[feature]["input_size"]

    model = model_class(
        num_classes=num_classes,
        backbone=backbone,
        input_size=input_size,
        multi_label=multi_label,
    )

    return model


def load_pipnet(config, state_dict, model_class):
    model = create_pipnet(
        feature=config["features"],
        pretrained=config["init_feature_weights"],
        num_classes=config["num_classes"],
        multi_label=config["multi_label"],
        model_class=model_class,
    )

    state_dict = torch.load(state_dict)
    model.load_state_dict(state_dict, strict=False)

    return model


model_creation_fn_dict = {
    "protopnet": create_protopnet,
    "protopool": create_protopool,
    "pipnet": create_pipnet,
    "protomask": create_protomask,
}

model_loading_fn_dict = {
    "protopnet": load_protopnet,
    "protopool": load_protopool,
    "pipnet": load_pipnet,
    "protomask": load_protomask,
}
