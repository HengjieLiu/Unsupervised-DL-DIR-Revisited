# deformation_decoder/building_blocks/deformation_prediction_block_vfa_standard.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from ..registries import register_deformation_decoder_block
# from .block_utils import get_activation, get_normalization

### VFA related imports
from .vfa_utils import Attention
from .vfa_utils import identity_grid_like  # for generating identity grids
# from vfa.utils.utils import identity_grid_like  # for generating identity grids

# import utils_warp  # which provides: dvf_upsample, SpatialTransformer, ComposeDVF



@register_deformation_decoder_block("dp_vfa_std")
class DPBlockVFAStandard(nn.Module):
    """
    A deformation prediction block that uses attention (in the spirit of VFA)
    to compute a residual displacement field (ddf) from a pair of fused feature maps.
    
    The inputs are:
      - feat_moving: warped moving image feature map [B, C, H, W, D]
      - feat_fixed:  fixed image feature map [B, C, H, W, D]
      
    The block first projects each feature map into an embedding space,
    tokenizes the fixed feature (to form queries Q) and the moving feature (to form keys K),
    and then uses an attention mechanism to “retrieve” a local displacement from a precomputed radial vector field R.
    
    Finally, the output is converted to the ddf convention by subtracting the identity grid.
    """
    def __init__(self, spatial_size, in_channels, attn_embed_channels=16, beta=1.0, temperature=None, similarity='inner_product', no_proj=False):
        """
        spatial_size: tuple (H, W, D) for the current level.
        in_channels: number of channels from the concatenated features (typically 1 * n_input_features_per_level)
        attn_embed_channels: number of channels in the embedding (can be configured)
        beta: scaling factor (from VFA, e.g. 1.0)
        temperature: if None, will be computed as sqrt(attn_embed_channels)
        """
        super().__init__()

        self.dim = len(spatial_size)
        
        # We use small convolutional layers to project both inputs into an embedding space.
        # (Note: these can be made non-trainable if desired.)
        # self.proj_fixed = nn.Conv3d(in_channels, attn_embed_channels, kernel_size=3, padding=1)
        # self.proj_moving = nn.Conv3d(in_channels, attn_embed_channels, kernel_size=3, padding=1)

        self.no_proj = no_proj
        if not self.no_proj:
            self.proj = nn.Conv3d(in_channels, attn_embed_channels, kernel_size=3, padding=1)
        
        # Initialize the attention module.
        self.attention = Attention()
        self.beta = beta # right now we don't make this trainable
        # self.beta = nn.Parameter(torch.tensor([float(initialize)]))

        self.similarity = similarity
        self.temperature = temperature
        
        # Precompute the radial vector field R.
        # In VFA, R is computed from a normalized identity grid.
        # We simulate this by creating a dummy tensor with spatial_size and then tokenizing it.
        # Use normalize=True to get values in a canonical range.
        # Tokenize: flatten spatial dimensions and transpose so that R becomes [N, 3]
        
        r = identity_grid_like(
            torch.zeros(1, 1, *[3 for _ in range(self.dim)]),
            normalize=True
        ) # [1, 3, 3, 3, 3]
        # self.R = self.to_token(r).squeeze().detach() # [27, 3]
        r_token = self.to_token(r).squeeze().detach() # [27, 3]
        self.register_buffer('R', r_token) # will automatically be moved to the appropriate device when you call .to(device) on your model
        
        # Optionally, freeze the parameters (making VFA non‐trainable)
        # for param in self.parameters():
        #     param.requires_grad = False

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
        # Assume inputs are from the two feature lists.
        # Each has shape [B, C, H, W, D] where C is the number of channels per input.
        # They are the outputs from your feature fusion (typically coming from concatenation)
        B, C, H, W, D = feat_fixed.shape
        
        # Split the concatenated input (if necessary) into the two modalities.
        # Here we assume that the moving and fixed features were concatenated along channel dim.
        # (Alternatively, if you already provided them separately, adjust accordingly.)
        # In our design, feat_fixed and feat_moving are provided separately.
        if not self.no_proj:
            fixed_emb = self.proj(feat_fixed)   # shape: [B, attn_embed_channels, H, W, D]
        else:
            fixed_emb = feat_fixed
        
        # pad moving features
        pad_size = [1 for _ in range(self.dim * 2)]
        feat_moving = nnf.pad(feat_moving, pad=tuple(pad_size), mode='replicate')

        if not self.no_proj:
            moving_emb = self.proj(feat_moving)  # shape: [B, attn_embed_channels, H, W, D]
        else:
            moving_emb = feat_moving
        
        if self.similarity == 'cosine':
            fixed_emb = nnf.normalize(fixed_emb, dim=1)
            moving_emb = nnf.normalize(moving_emb, dim=1)
        
        # Tokenize the fixed features: reshape to [B, N, attn_embed_channels] where N=H*W*D,
        # then add a singleton dimension for “patch” (set to 1 here)
        # Q = fixed_emb.permute(0,2,3,4,1).reshape(B, H*W*D, -1).unsqueeze(2)  # [B, N, 1, attn_embed_channels]
        permute_order = [0] + list(range(2, 2 + self.dim)) + [1]
        Q = fixed_emb.permute(*permute_order).unsqueeze(-2)
        
        # Tokenize the moving features similarly to form K.
        # K = moving_emb.permute(0,2,3,4,1).reshape(B, H*W*D, -1).unsqueeze(2)  # [B, N, 1, attn_embed_channels]
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

    

