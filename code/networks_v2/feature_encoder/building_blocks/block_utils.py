# feature_encoder/building_blocks/block_utils.py

import torch.nn as nn

def get_activation(act_name):
    act, params = act_name
    if act.lower() == 'relu':
        return nn.ReLU(**params)
    elif act.lower() == 'leakyrelu':
        return nn.LeakyReLU(**params)
    elif act.lower() == 'prelu':
        return nn.PReLU(**params)
    else:
        raise ValueError(f"Unsupported activation: {act}")

def get_normalization(norm_name, num_features):
    if norm_name is None:
        return None
    norm, params = norm_name
    if norm.lower() == 'group':
        return nn.GroupNorm(num_groups=params.get('num_groups', 4), num_channels=num_features, affine=params.get('affine', True))
    elif norm.lower() == 'instance':
        return nn.InstanceNorm3d(num_features, affine=params.get('affine', True))
    else:
        raise ValueError(f"Unsupported normalization: {norm}")

def depthpoint_conv(in_channels, out_channels, kernel_size, stride, padding, bias):
    depth_conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride=stride,
                           padding=padding, groups=in_channels, bias=bias)
    point_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                           stride=1, padding=0, bias=bias)
    return nn.Sequential(depth_conv, point_conv)
