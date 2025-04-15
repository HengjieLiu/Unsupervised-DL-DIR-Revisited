# deformation_decoder/building_blocks/deformation_prediction_vfa_block_conv_standard.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from ..registries import register_deformation_decoder_block
from .block_utils import get_activation, get_normalization

### VFA related imports
from .vfa_utils import Attention
from .vfa_utils import identity_grid_like  # for generating identity grids
# from vfa.utils.utils import identity_grid_like  # for generating identity grids

# import utils_warp  # which provides: dvf_upsample, SpatialTransformer, ComposeDVF



@register_deformation_decoder_block("dp_vfa_conv")
class DPVFAConvBlock(nn.Module):
    """
    Deformation prediction block that combines a convolutional projection head with a final
    attention-based displacement prediction, in the spirit of VFA.
    
    Instead of concatenating moving and fixed features, this block processes them separately:
      - For each branch (fixed and moving), a residual convolution block (by default with n_convolutions=2)
        processes the input features (with in_channels given by n_input_features_per_level, not multiplied by 2)
        to produce an intermediate representation. The output channel size of these blocks is given by
        `conv_out_channels` (typically taken from your list n_features_per_level).
      - A final 1×1×1 convolution (applied separately to each branch) projects the output into an embedding
        space with channel number `attn_embed_channels` (to be provided via a new list, e.g. n_attention_per_level).
      - The fixed branch is tokenized to yield query vectors Q while the moving branch is tokenized (or
        “unfolded”) to yield keys K. An attention module then computes a weighted displacement from a
        pre‐computed radial field R.
      - Finally, the predicted displacement is scaled (by beta), added to an identity grid, and then the
        ddf is computed by subtracting the identity grid.
    
    Parameters:
      - spatial_size: tuple (H, W, D) for the current level.
      - in_channels: number of channels of each input feature map (from the fixed and moving sides separately).
      - conv_out_channels: number of output channels of the residual conv block (from n_features_per_level).
      - attn_embed_channels: number of channels for the final 1×1×1 projection (from n_attention_per_level).
      - kernel_size, bias, n_convolutions, res_skip, act_name, norm_name: control the convolutional block
        (default settings similar to DPBlockConvBlockStandard).
      - beta: scaling factor for the displacement (default 1.0).
      - temperature: temperature for attention scaling (if None, set to sqrt(attn_embed_channels)).
      - similarity: similarity metric to use in attention (default 'inner_product'; if set to 'cosine', normalization
        is applied).
    """
    def __init__(self, spatial_size, in_channels, conv_out_channels, attn_embed_channels=16, 
                 kernel_size=3, bias=True, n_convolutions=2, res_skip=True, 
                 act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
                 norm_name=("group", {"num_groups": 4, "affine": True}),
                 beta=1.0, temperature=None, similarity='inner_product'):
        super().__init__()
        
        self.dim = len(spatial_size)
        
        # Process fixed and moving features separately via shared residual conv blocks.
        self.conv_block = self._build_residual_block(in_channels, conv_out_channels, kernel_size, bias,
                                                     n_convolutions, res_skip, act_name, norm_name)
        # Final 1x1x1 projection layers.
        self.proj = nn.Conv3d(conv_out_channels, attn_embed_channels, kernel_size=1, bias=bias)

        # Attention module.
        self.attention = Attention()
        self.beta = beta # right now we don't make this trainable
        # self.beta = nn.Parameter(torch.tensor([float(initialize)]))
        
        self.similarity = similarity
        self.temperature = temperature 
        # self.temperature = temperature if temperature is not None else math.sqrt(attn_embed_channels)

         # Precompute the radial vector field R.
        r = identity_grid_like(
            torch.zeros(1, 1, *[3 for _ in range(self.dim)]),
            normalize=True
        ) # [1, 3, 3, 3, 3]
        # self.R = self.to_token(r).squeeze().detach() # [27, 3]
        r_token = self.to_token(r).squeeze().detach() # [27, 3]
        self.register_buffer('R', r_token) # will automatically be moved to the appropriate device when you call .to(device) on your model
    
    def _build_residual_block(self, in_channels, out_channels, kernel_size, bias, n_convolutions, res_skip, act_name, norm_name):
        padding = kernel_size // 2
        conv_layers = []
        for i in range(n_convolutions):
            conv = nn.Conv3d(in_channels if i == 0 else out_channels,
                             out_channels,
                             kernel_size,
                             stride=1,
                             padding=padding,
                             bias=bias)
            layers = [conv]
            if norm_name is not None:
                layers.append(get_normalization(norm_name, out_channels))
            if i < n_convolutions - 1:
                layers.append(get_activation(act_name))
            conv_layers.append(nn.Sequential(*layers))
        main_branch = nn.Sequential(*conv_layers)
        post_activation = get_activation(act_name)
        if res_skip:
            skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
            skip_norm = get_normalization(norm_name, out_channels) if norm_name is not None else None
        else:
            skip_conv = None
            skip_norm = None
        return _ResidualBlock(main_branch, skip_conv, skip_norm, post_activation)
    
    def to_token(self, x):
        # x: [B, 1, H, W, D] --> [B, H*W*D, 1]
        x = x.flatten(start_dim=2)  # [B, 1, H*W*D]
        x = x.transpose(-1, -2)     # [B, H*W*D, 1]
        return x

    def get_candidate_from_tensor(self, x, dim, kernel=3, stride=1):
        if dim == 3:
            '''from tensor with [Batch x Feature x Height x Weight x Depth],
                    extract patches [Batch x Feature x HxWxD x Patch],
                    and reshape to [Batch x HxWxS x Patch x Feature]'''
            patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride).unfold(4, kernel, stride)
            patches = patches.flatten(start_dim=5)
            token = patches.permute(0, 2, 3, 4, 5, 1)
        elif dim == 2:
            '''From tensor with [Batch x Feature x Height x Weight],
                    extract patches [Batch x Feature x HxW x Patch],
                    and reshape to [Batch x HxW x Patch x Feature]'''
            patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
            patches = patches.flatten(start_dim=4)
            token = patches.permute(0, 2, 3, 4, 1)

        return token
    
    def forward(self, feat_moving, feat_fixed):

        feat_fixed  = self.conv_block(feat_fixed)
        feat_moving = self.conv_block(feat_moving)
        
        # Split the concatenated input (if necessary) into the two modalities.
        # Here we assume that the moving and fixed features were concatenated along channel dim.
        # (Alternatively, if you already provided them separately, adjust accordingly.)
        # In our design, feat_fixed and feat_moving are provided separately.
        fixed_emb  = self.proj(feat_fixed)   # shape: [B, embed_channels, H, W, D]
        
        # pad moving features
        pad_size = [1 for _ in range(self.dim * 2)]
        feat_moving = nnf.pad(feat_moving, pad=tuple(pad_size), mode='replicate')
        
        moving_emb = self.proj(feat_moving)  # shape: [B, embed_channels, H, W, D]
        
        if self.similarity == 'cosine':
            fixed_emb = nnf.normalize(fixed_emb, dim=1)
            moving_emb = nnf.normalize(moving_emb, dim=1)
        
        # Tokenize the fixed features: reshape to [B, N, embed_channels] where N=H*W*D,
        # then add a singleton dimension for “patch” (set to 1 here)
        # Q = fixed_emb.permute(0,2,3,4,1).reshape(B, H*W*D, -1).unsqueeze(2)  # [B, N, 1, embed_channels]
        permute_order = [0] + list(range(2, 2 + self.dim)) + [1]
        Q = fixed_emb.permute(*permute_order).unsqueeze(-2)
        
        # Tokenize the moving features similarly to form K.
        # K = moving_emb.permute(0,2,3,4,1).reshape(B, H*W*D, -1).unsqueeze(2)  # [B, N, 1, embed_channels]
        K = self.get_candidate_from_tensor(moving_emb, self.dim)

        # feature matching and location retrieval
        # print('Q.shape: ', Q.shape)           # example shapes: Q.shape:  torch.Size([1, 40, 56, 48, 1, 32])
        # print('K.shape: ', K.shape)           # example shapes: K.shape:  torch.Size([1, 40, 56, 48, 27, 32])
        # print('self.R.shape: ', self.R.shape) # example shapes: self.R.shape:  torch.Size([27, 3])
        
        local_disp = self.attention(Q, K, self.R, self.temperature)
        permute_order = [0, -1] + list(range(1, 1 + self.dim))
        local_disp = local_disp.squeeze(-2).permute(*permute_order)
        # identity_grid = identity_grid_like(local_disp, normalize=False)
        # local_grid = local_disp * self.beta / 2**self.int_steps + identity_grid
        
        # # Use attention to compute a weighted “local displacement”.
        # # (Note: in a full VFA implementation you might extract patches for K;
        # # here we simply use a patch size of 1.)
        # local_disp = self.attention(Q, K, self.R, self.temperature)  # [B, N, 1, 3]
        # # Reshape back to spatial dimensions: [B, 3, H, W, D]
        # local_disp = local_disp.squeeze(2).transpose(1,2).view(B, 3, H, W, D)
        
        # # In the VFA code the predicted grid is computed as:
        # #   local_grid = local_disp * beta + identity_grid
        # # and then the displacement field is ddf = composed_grid - identity_grid.
        # identity = identity_grid_like(local_disp, normalize=False)  # [B, 3, H, W, D]
        # composed_grid = local_disp * self.beta + identity
        # ddf_res = composed_grid - identity  # which is equivalent to local_disp * beta

        ddf_res = local_disp # the beta formulation is not used for now
        
        return ddf_res


###############################################################################
# A helper residual block (similar to that used in DPBlockConvBlockStandard)
###############################################################################
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
