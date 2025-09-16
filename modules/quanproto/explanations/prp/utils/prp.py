import numpy as np
from .lrp_general6 import *
from quanproto.explanations.prp.features.resnet_canonized import *

canonized_feature_dict = {
    "resnet18": resnet18_canonized,
    "resnet34": resnet34_canonized,
    "resnet50": resnet50_canonized,
    "resnet50ext": resnet50ext_canonized,
    "resnet101": resnet101_canonized,
    "resnet152": resnet152_canonized,
}

lrp_params_def1 = {
    "conv2d_ignorebias": True,
    "eltwise_eps": 1e-6,
    "linear_eps": 1e-6,
    "pooling_eps": 1e-6,
    "use_zbeta": True,
}

lrp_layer2method = {
    "nn.ReLU": relu_wrapper_fct,
    "nn.Sigmoid": sigmoid_wrapper_fct,
    "nn.Softmax": softmax_wrapper_fct,
    "nn.BatchNorm2d": relu_wrapper_fct,
    "nn.Conv2d": conv2d_beta0_wrapper_fct,
    "nn.Linear": linearlayer_eps_wrapper_fct,
    "nn.AdaptiveAvgPool2d": adaptiveavgpool2d_wrapper_fct,
    "nn.AdaptiveMaxPool2d": adaptivemaxpool2d_wrapper_fct,
    "nn.MaxPool2d": maxpool2d_wrapper_fct,
    "sum_stacked2": eltwisesum_stacked2_eps_wrapper_fct,
}
