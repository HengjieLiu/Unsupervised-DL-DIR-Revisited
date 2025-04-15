# feature_encoder/building_blocks/conv_block_depthpoint_maxpool_down_with_skip

import torch
import torch.nn as nn
from ..registries import register_block

from .block_utils import get_activation, get_normalization, depthpoint_conv


@register_block('convblock_depthpoint_maxpool')
class ConvBlockDepthPointMaxPoolDownWithSkip(nn.Module):
    """
    Depthwise-pointwise convolution block with residual skip and max pooling based downsampling.
    
    If downsample is True, input is downsampled using max pooling before applying the depthpoint conv block.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        n_convolutions: int = 2,
        downsample: bool = False,
        res_skip: bool = True,
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
        norm_name=("group", {"num_groups": 4, "affine": True}),
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.res_skip = res_skip
        padding = kernel_size // 2
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) if downsample else None
        
        conv_layers = []
        for i in range(n_convolutions):
            conv = depthpoint_conv(in_channels if i==0 else out_channels, out_channels,
                                   kernel_size, stride=1, padding=padding, bias=bias)
            layers = [conv]
            if norm_name is not None:
                layers.append(get_normalization(norm_name, out_channels))
            if i < n_convolutions - 1:
                layers.append(get_activation(act_name))
            conv_layers.append(nn.Sequential(*layers))
        self.main_branch = nn.Sequential(*conv_layers)
        self.post_activation = get_activation(act_name)
        
        if self.res_skip:
            self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                       stride=1, padding=0, bias=bias)
            self.skip_norm = get_normalization(norm_name, out_channels) if norm_name is not None else None
    
    def forward(self, x):
        if self.downsample and self.pool is not None:
            x = self.pool(x)
        out = self.main_branch(x)
        if self.res_skip:
            residual = self.skip_conv(x)
            if self.skip_norm is not None:
                residual = self.skip_norm(residual)
            out = out + residual
        out = self.post_activation(out)
        return out
