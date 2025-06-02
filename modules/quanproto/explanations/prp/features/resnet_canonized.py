"""
This code was copied from the following repository:
Reference: https://github.com/romain-xu-darme/prototype_sanity_checks
"""

from quanproto.features.resnet_features import ResNet_features, BasicBlock, Bottleneck
from quanproto.explanations.prp.utils.lrp_general6 import *


class Modulenotfounderror(Exception):
    pass


class Bottleneck_fused(Bottleneck):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_fused, self).__init__(inplanes, planes, stride, downsample)

        # own
        self.elt = sum_stacked2()  # eltwisesum2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = self.elt(torch.stack([out, identity], dim=0))  # self.elt(out,identity)
        out = self.relu(out)

        return out


class BasicBlock_fused(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_fused, self).__init__(inplanes, planes, stride, downsample)

        # own
        self.elt = sum_stacked2()  # eltwisesum2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        # out = self.relu(out)

        out = self.elt(torch.stack([out, identity], dim=0))  # self.elt(out,identity)
        out = self.relu(out)

        return out


class ResNet_canonized(ResNet_features):

    def __init__(self, block, layers, strides=[1, 2, 2, 2], zero_init_residual=False):
        super(ResNet_canonized, self).__init__(
            block, layers, strides=strides, zero_init_residual=zero_init_residual
        )

        ######################
        # change
        ######################
        # own
        # self.avgpool = nn.AvgPool2d(kernel_size=7,stride=7 ) #nn.AdaptiveAvgPool2d((1, 1))

    # runs in your current module to find the object layer3.1.conv2, and replaces it by the obkect stored in value (see         success=iteratset(self,components,value) as initializer, can be modified to run in another class when replacing that self)
    def setbyname(self, name, value):

        def iteratset(obj, components, value):

            if not hasattr(obj, components[0]):
                return False
            elif len(components) == 1:
                setattr(obj, components[0], value)
                # print('found!!', components[0])
                # exit()
                return True
            else:
                nextobj = getattr(obj, components[0])
                return iteratset(nextobj, components[1:], value)

        components = name.split(".")
        success = iteratset(self, components, value)
        return success

    def copy_weights(self, net, lrp_params, lrp_layer2method):
        # assert( isinstance(net,ResNet))

        # --copy linear
        # --copy conv2, while fusing bns
        # --reset bn

        # first conv, then bn,
        # means: when encounter bn, find the conv before -- implementation dependent

        updated_layers_names = []

        last_src_module_name = None
        last_src_module = None

        for src_module_name, src_module in net.named_modules():
            # print("at src_module_name", src_module_name)

            foundsth = False

            if isinstance(src_module, nn.Linear):
                # copy linear layers
                foundsth = True
                # print("is Linear")
                # m =  oneparam_wrapper_class( copy.deepcopy(src_module) , linearlayer_eps_wrapper_fct(), parameter1 = linear_eps )
                wrapped = get_lrpwrapperformodule(
                    copy.deepcopy(src_module), lrp_params, lrp_layer2method
                )
                # print(wrapped)
                # exit()
                if False == self.setbyname(src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + src_module_name + " in target net to copy"
                    )
                updated_layers_names.append(src_module_name)
            # end of if

            if isinstance(src_module, nn.Conv2d):
                # store conv2d layers
                foundsth = True
                # print("is Conv2d")
                last_src_module_name = src_module_name
                last_src_module = src_module
            # end of if

            if isinstance(src_module, nn.BatchNorm2d):
                # conv-bn chain
                foundsth = True
                # print("is BatchNorm2d")

                if (True == lrp_params["use_zbeta"]) and (last_src_module_name == "conv1"):
                    thisis_inputconv_andiwant_zbeta = True
                else:
                    thisis_inputconv_andiwant_zbeta = False

                m = copy.deepcopy(last_src_module)
                m = bnafterconv_overwrite_intoconv(m, bn=src_module)
                # wrap conv
                wrapped = get_lrpwrapperformodule(
                    m,
                    lrp_params,
                    lrp_layer2method,
                    thisis_inputconv_andiwant_zbeta=thisis_inputconv_andiwant_zbeta,
                )
                # print(wrapped)
                # exit()

                if False == self.setbyname(last_src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + last_src_module_name + " in target net to copy"
                    )

                updated_layers_names.append(last_src_module_name)

                # wrap batchnorm
                wrapped = get_lrpwrapperformodule(resetbn(src_module), lrp_params, lrp_layer2method)
                # print(wrapped)
                # exit()
                if False == self.setbyname(src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + src_module_name + " in target net to copy"
                    )
                updated_layers_names.append(src_module_name)
            # end of if

            # if False== foundsth:
            #  print('!untreated layer')
            # print("\n")

        # sum_stacked2 is present only in the targetclass, so must iterate here
        for target_module_name, target_module in self.named_modules():

            if isinstance(target_module, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                # print(wrapped)
                # exit()
                if False == self.setbyname(target_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + src_module_name + " in target net to copy"
                    )
                updated_layers_names.append(target_module_name)

            if isinstance(target_module, sum_stacked2):

                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                # print(wrapped)
                # exit()
                if False == self.setbyname(target_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module "
                        + target_module_name
                        + " in target net , impossible!"
                    )
                updated_layers_names.append(target_module_name)

        # this prints the module and Sequential, but not the layers
        for target_module_name, target_module in self.named_modules():
            if target_module_name not in updated_layers_names:
                # print("not updated:", target_module_name)
                pass


def _resnet_canonized(arch, block, layers, **kwargs):
    model = ResNet_canonized(block, layers, **kwargs)
    # if pretrained:
    #     raise Cannotloadmodelweightserror("explainable nn model wrapper was never meant to load dictionary weights, load into standard model first, then instatiate this class from the standard model")
    return model


def resnet18_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized(
        "resnet18", BasicBlock_fused, [2, 2, 2, 2], pretrained, progress, **kwargs
    )


def resnet50_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized("resnet50", Bottleneck_fused, [3, 4, 6, 3], **kwargs)


def resnet34_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized("resnet34", BasicBlock_fused, [3, 4, 6, 3], **kwargs)


def resnet152_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized("resnet152", Bottleneck_fused, [3, 8, 36, 3], **kwargs)


def resnet101_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized("resnet101", Bottleneck_fused, [3, 4, 23, 3], **kwargs)
