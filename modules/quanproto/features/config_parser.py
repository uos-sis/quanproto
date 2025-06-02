"""
This module contains the configuration for the feature extraction models.
"""

import quanproto.features.densenet_features as densenet
import quanproto.features.efficientnet_features as efficientnet
import quanproto.features.resnet_features as resnet
import quanproto.features.vgg_features as vgg

IMAGENET_DEFAULT_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

IMAGENET_STANDARD_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STANDARD_STD: tuple[float, float, float] = (0.47853944, 0.4732864, 0.47434163)

# TODO: add a possibility to use custom mean and std

feature_dict = {
    "vgg11": {
        "model_fn": vgg.vgg11_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "vgg11_bn": {
        "model_fn": vgg.vgg11_bn_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "vgg13": {
        "model_fn": vgg.vgg13_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "vgg13_bn": {
        "model_fn": vgg.vgg13_bn_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "vgg16": {
        "model_fn": vgg.vgg16_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "vgg16_bn": {
        "model_fn": vgg.vgg16_bn_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "vgg19": {
        "model_fn": vgg.vgg19_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "vgg19_bn": {
        "model_fn": vgg.vgg19_bn_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "resnet18": {
        "model_fn": resnet.resnet18_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "resnet18ext": {
        "model_fn": resnet.resnet18_ext_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "resnet34": {
        "model_fn": resnet.resnet34_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "resnet34ext": {
        "model_fn": resnet.resnet34_ext_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "resnet50": {
        "model_fn": resnet.resnet50_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "resnet50ext": {
        "model_fn": resnet.resnet50_ext_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "resnet101": {
        "model_fn": resnet.resnet101_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "resnet101ext": {
        "model_fn": resnet.resnet101_ext_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "resnet152": {
        "model_fn": resnet.resnet152_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "resnet152ext": {
        "model_fn": resnet.resnet152_ext_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "densenet121": {
        "model_fn": densenet.densenet121_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "densenet161": {
        "model_fn": densenet.densenet161_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "densenet169": {
        "model_fn": densenet.densenet169_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "densenet201": {
        "model_fn": densenet.densenet201_features,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "input_size": (3, 224, 224),
    },
    "efficientnet-b0": {
        "model_fn": efficientnet.efficientnet_b0_features,
        "mean": IMAGENET_STANDARD_MEAN,
        "std": IMAGENET_STANDARD_STD,
        "input_size": (3, 224, 224),
    },
}
