import torch.nn.functional as F


def min_pool2d(xs, **kwargs):
    return -F.max_pool2d(-xs, **kwargs)
