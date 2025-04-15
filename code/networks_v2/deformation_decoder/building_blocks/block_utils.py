# deformation_decoder/building_blocks/block_utils.py

import torch
import torch.nn as nn
from torch.nn import init

def get_activation(act_name):
    act, params = act_name
    if act.lower() == "relu":
        return nn.ReLU(**params)
    elif act.lower() == "leakyrelu":
        return nn.LeakyReLU(**params)
    elif act.lower() == "prelu":
        return nn.PReLU(**params)
    else:
        raise ValueError(f"Unsupported activation: {act}")

def get_normalization(norm_name, num_features):
    if norm_name is None:
        return None
    norm, params = norm_name
    if norm.lower() == "group":
        return nn.GroupNorm(num_groups=params.get("num_groups", 4), num_channels=num_features, affine=params.get("affine", True))
    elif norm.lower() == "instance":
        return nn.InstanceNorm3d(num_features, affine=params.get("affine", True))
    else:
        raise ValueError(f"Unsupported normalization: {norm}")

class FlowConv(nn.Sequential):
    """
    A convolutional block for flow prediction that initializes weights using a small-flow method.
    Ref:
        https://github.com/BailiangJ/rethink-reg/blob/6fc0af1f04a707bddbcfb5246e09e295d0b3a8fe/models/networks/transmorph.py#L935
    ChatGPT:
        If we tried to call super().__init__ before creating and customizing conv3d, we wouldn't have the updated weights/bias ready.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
        )
        # Initialize weight with small-flow initialization: normal with mean=0, std=1e-5.
        conv3d.weight = nn.Parameter(init.normal_(torch.empty(conv3d.weight.shape), mean=0, std=1e-5))
        conv3d.bias = nn.Parameter(torch.zeros_like(conv3d.bias))
        super().__init__(conv3d)