# deformation_decoder/building_blocks/flowconv_family.py

import torch
import torch.nn as nn
import torch.nn.init as init

"""
In SEFlowConv and CBAMFlowConv
    To accomodate for the 3 channel DDF
    
    Default setting of:
        reduction=16
        bottleneck_channels = max(out_channels // reduction, 1)
    are modified to 
        reduction=1
        bottleneck_channels = max(out_channels // reduction, 3)


"""

class VanillaFlowConv(nn.Sequential):
    """
    Vanilla FlowConv: a single 3x3x3 convolution with smallflow initialization.
    A convolutional block for flow prediction that initializes weights using a small-flow method.
    Ref:
        https://github.com/BailiangJ/rethink-reg/blob/6fc0af1f04a707bddbcfb5246e09e295d0b3a8fe/models/networks/transmorph.py#L935
    ChatGPT:
        If we tried to call super().__init__ before creating and customizing conv3d, we wouldn't have the updated weights/bias ready.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        print("INFO: FlowConv (VanillaFlowConv) is called, no need to use specialized initialization for output conv layer")
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
        
class PerChannelFlowConv(nn.Module):
    """
    FlowConv variant that adds a trainable per-channel scaling after the convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        print("INFO: FlowConv (PerChannelFlowConv) is called, no need to use specialized initialization for output conv layer")
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
        )
        # Use smallflow initialization for the convolution.
        self.conv.weight = nn.Parameter(init.normal_(torch.empty(self.conv.weight.shape), mean=0, std=1e-5))
        self.conv.bias = nn.Parameter(torch.zeros_like(self.conv.bias))
        # One scalar per output channel.
        self.channel_scalars = nn.Parameter(torch.ones(out_channels))
        
    def forward(self, x):
        out = self.conv(x)
        return out * self.channel_scalars.view(1, -1, 1, 1, 1)

class SEFlowConv(nn.Module):
    """
    FlowConv with Squeeze-and-Excitation.
    
    The reduction parameter determines the bottleneck size in the SE block.
    For example, if out_channels=64 and reduction=16, then the bottleneck has 64//16=4 channels.
    Typical values for reduction range from 4 to 16.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, reduction=1):
        print("INFO: FlowConv (SEFlowConv) is called, no need to use specialized initialization for output conv layer")
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True
        )
        # Smallflow initialization.
        self.conv.weight = nn.Parameter(init.normal_(torch.empty(self.conv.weight.shape), mean=0, std=1e-5))
        self.conv.bias = nn.Parameter(torch.zeros_like(self.conv.bias))
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        bottleneck_channels = max(out_channels // reduction, 3)
        self.fc1 = nn.Conv3d(out_channels, bottleneck_channels, kernel_size=1, bias=True)
        self.fc2 = nn.Conv3d(bottleneck_channels, out_channels, kernel_size=1, bias=True)
        # You can also initialize fc layers here if desired.
        nn.init.kaiming_normal_(self.fc1.weight, a=0.2)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, a=0.2)
        nn.init.constant_(self.fc2.bias, 0)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.conv(x)
        w = self.global_pool(out)
        w = self.fc1(w)
        w = nn.functional.relu(w, inplace=True)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return out * w

class CBAMFlowConv(nn.Module):
    """
    FlowConv with CBAM-style channel attention.
    
    In this variant, a shared MLP (applied to both the average-pooled and max-pooled descriptors)
    computes attention weights. The reduction parameter here controls the bottleneck size,
    similar to the SE block.
    
    Typical values for reduction are in the range 4-16.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, reduction=1):
        print("INFO: FlowConv (CBAMFlowConv) is called, no need to use specialized initialization for output conv layer")
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True
        )
        # Smallflow initialization.
        self.conv.weight = nn.Parameter(init.normal_(torch.empty(self.conv.weight.shape), mean=0, std=1e-5))
        self.conv.bias = nn.Parameter(torch.zeros_like(self.conv.bias))
        
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        bottleneck_channels = max(out_channels // reduction, 3)
        self.mlp = nn.Sequential(
            nn.Conv3d(out_channels, bottleneck_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck_channels, out_channels, kernel_size=1, bias=True)
        )
        # Initialize MLP layers (could also use smallflow here if desired).
        nn.init.kaiming_normal_(self.mlp[0].weight, a=0.2)
        nn.init.constant_(self.mlp[0].bias, 0)
        nn.init.kaiming_normal_(self.mlp[2].weight, a=0.2)
        nn.init.constant_(self.mlp[2].bias, 0)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.conv(x)
        avg_pool = self.global_avg_pool(out)
        max_pool = self.global_max_pool(out)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        attn = self.sigmoid(avg_out + max_out)
        return out * attn
