# deformation_decoder/building_blocks/deformation_prediction_block_conv_block_standard.py


from typing import Optional, Union, Tuple, Dict, Any

import torch
import torch.nn as nn

from ..registries import register_deformation_decoder_block
from .block_utils import get_activation, get_normalization

# from .block_utils import FlowConv
from ..building_blocks.flowconv_family import VanillaFlowConv, PerChannelFlowConv, SEFlowConv, CBAMFlowConv


@register_deformation_decoder_block("dp_convblock_std")
class DPBlockConvBlockStandard(nn.Module):
    """
    Basic deformation prediction block using a residual convolution block.
    
    This block concatenates two inputs along the channel dimension (e.g. warped moving and fixed features),
    applies a residual convolution block (without downsampling), and then produces a residual DDF via an output block.
    The output block uses FlowConv (if use_flowconv is True) so that its weights are initialized with the small-flow method.
    
    Parameters:
        in_channels: number of channels of the concatenated input (should equal n_input_features_per_level[i]*2).
        out_channels: number of decoder channels at this level (n_features_per_level[i]).
        kernel_size, bias, n_convolutions, res_skip, act_name, norm_name: control the residual block.
        use_flowconv: if True, use FlowConv for the output block (default True).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        n_convolutions: int = 2,
        res_skip: bool = True,
        act_name: Tuple[str, Dict[str, Any]] = ("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
        norm_name: Tuple[str, Dict[str, Any]] = ("group", {"num_groups": 4, "affine": True}),
        use_flowconv: bool = True,
        flowconv_type: Optional[str] = None,
    ):
        super().__init__()
        
        self.res_block = self._build_residual_block(
            in_channels, out_channels, kernel_size, bias, n_convolutions, res_skip, act_name, norm_name
        )
        
        if flowconv_type is None or flowconv_type == "vanilla":
            FlowConvVariant = VanillaFlowConv
        elif flowconv_type == "per_channel":
            FlowConvVariant = PerChannelFlowConv
        elif flowconv_type == "se":
            FlowConvVariant = SEFlowConv
        elif flowconv_type == "cbam":
            FlowConvVariant = CBAMFlowConv
        else:
            raise ValueError(f"Unsupported flowconv_type: {flowconv_type}")
        
        # The output block: use FlowConv if specified; otherwise, use a standard 3x3 convolution.
        # The different variants do not seem to make a difference
        if use_flowconv:
            # self.out_block = FlowConv(out_channels, 3, kernel_size=3)
            self.out_block = FlowConvVariant(out_channels, 3, kernel_size=3)
        else:
            self.out_block = nn.Conv3d(out_channels, 3, kernel_size=3, stride=1, padding=1, bias=bias)
    
    def _build_residual_block(self, in_channels, out_channels, kernel_size, bias, n_convolutions, res_skip, act_name, norm_name):
        padding = kernel_size // 2
        conv_layers = []
        # Always use stride=1 since no downsampling is needed.
        for i in range(n_convolutions):
            conv = nn.Conv3d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                bias=bias
            )
            layers = [conv]
            if norm_name is not None:
                layers.append(get_normalization(norm_name, out_channels))
            # Apply activation for all but the last convolution.
            if i < n_convolutions - 1:
                layers.append(get_activation(act_name))
            conv_layers.append(nn.Sequential(*layers))
        main_branch = nn.Sequential(*conv_layers)
        self.post_activation = get_activation(act_name)
        
        if res_skip:
            skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
            skip_norm = get_normalization(norm_name, out_channels) if norm_name is not None else None
        else:
            skip_conv = None
            skip_norm = None
        
        return _ResidualBlock(main_branch, skip_conv, skip_norm, self.post_activation)

    ### outdated as of 04/04/25
    # def forward(self, x1, x2):
    #     # Concatenate the two inputs along the channel dimension.
    #     x = torch.cat([x1, x2], dim=1)
    #     x = self.res_block(x)
    #     out = self.out_block(x)
    #     return out
    
    def forward(self, x):
        # Assume input is already concatenated along the channel dimension.
        x = self.res_block(x)
        out = self.out_block(x)
        return out

class _ResidualBlock(nn.Module):
    def __init__(self, main_branch, skip_conv, skip_norm, post_activation):
        super().__init__()
        self.main_branch = main_branch
        self.skip_conv = skip_conv
        self.skip_norm = skip_norm
        self.post_activation = post_activation
        
    def forward(self, x):
        out = self.main_branch(x)
        if self.skip_conv is not None:
            residual = self.skip_conv(x)
            if self.skip_norm is not None:
                residual = self.skip_norm(residual)
            out = out + residual
        out = self.post_activation(out)
        return out

