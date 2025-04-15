# deformation_decoder/decoders/deformation_decoder_pyramidal_vfa.py

import torch
import torch.nn as nn
import torch.nn.init as init
        
from typing import Sequence

from ..registries import register_deformation_decoder, DEFORMATION_DECODER_BLOCK_REGISTRY

# from ..building_blocks.deformation_prediction_block_vfa_standard import DPBlockVFAStandard

import sys
sys.path.insert(0, '/homebase/DL_projects/wavereg/code')
import utils_warp  # Assumed to provide SpatialTransformer, ComposeDVF, dvf_upsample

# make this global for readability
dp_block_filtered_keys = [
    'kernel_size', 'bias', 
    'n_convolutions', 'res_skip', 
    'act_name', 'norm_name', 
    'beta', 'temperature', 'similarity', 'no_proj',
]


###############################################################################
# New decoder: DeformationDecoderPyramidalVFA
###############################################################################
# This class follows the structure of your existing DeformationDecoderPyramidalCNN,
# but replaces the convolution‐based DP blocks with our DPBlockVFAStandard.
# It expects two lists of multi‐scale features (from moving and fixed images)
# and produces a final displacement field (ddf) in the same format.
#
# Note: The VFA convention is to output a composed grid, but here we always convert to the
# ddf convention (ddf = composed_grid - identity) so that both ddf_res and ddf_tot are tracked.
###############################################################################
@register_deformation_decoder("pyramidal_vfa_std")
class DeformationDecoderPyramidalVFA(nn.Module):
    def __init__(self, **decoder_params):
        """
        Expected keys in decoder_params:
          - img_size: tuple (H, W, D)
          - n_output_channels: typically 3 for 3D
          - n_levels: number of pyramid levels
          - n_input_features_per_level: list of input channel counts (for each level)
          - n_features_per_level: list of decoder channels for each level (used for DPBlockVFAStandard)
          - dp_block_type: registry key for the DP block to use (e.g. "dp_vfa_std" or "dp_vfa_conv")
                - For dp_vfa_conv: additional lists such as "n_attention_per_level" must be provided.
          - Plus any common DP block parameters: e.g. 'kernel_size', 'bias', 'n_convolutions', 'res_skip', 'act_name', 'norm_name', 'beta', 'temperature', 'similarity'.
        """
        super().__init__()

        # Extract parameters from decoder_params
        self.img_size = decoder_params["img_size"]
        self.n_output_channels = decoder_params["n_output_channels"]
        self.n_levels = decoder_params["n_levels"]
        self.n_input_features_per_level = decoder_params["n_input_features_per_level"]
        self.n_features_per_level = decoder_params.get("n_features_per_level", None)
        self.n_attention_per_level = decoder_params.get("n_attention_per_level", None)
        # for VFA projection layer
        print('INFO in DeformationDecoderPyramidalVFA')
        print('self.n_input_features_per_level: ', self.n_input_features_per_level)
        print('self.n_features_per_level: ', self.n_features_per_level)
        print('self.n_attention_per_level: ', self.n_attention_per_level)
        
        # Upsampling function for displacement fields.
        # self.upsample_ddf = utils_warp.dvf_upsample
        self.upsample_ddf = utils_warp.ResizeFlow(spatial_scale=2, flow_scale=2, ndim=3)
        
        # Initialize transformers and composers for coarse-to-fine composition.
        self.transformers = nn.ModuleList()
        self.composers = nn.ModuleList()
        for i in range(self.n_levels):
            # For level i, the spatial size is img_size divided by 2^(n_levels-1 - i)
            spatial_size = tuple(s // (2 ** (self.n_levels - 1 - i)) for s in self.img_size)
            self.transformers.append(utils_warp.SpatialTransformer(spatial_size))
            self.composers.append(utils_warp.ComposeDVF(spatial_size))
        
        # --- DP Block instantiation ---
        # Get the dp block type from decoder_params; default to "dp_vfa_std"
        dp_block_type = decoder_params.get("dp_block_type", "dp_vfa_std")
        dp_block_cls = DEFORMATION_DECODER_BLOCK_REGISTRY.get(dp_block_type)
        if dp_block_cls is None:
            raise ValueError(f"Unsupported dp block type: {dp_block_type}")
        
        # Filter common DP block keys.
        dp_block_kwargs = {k: decoder_params[k] for k in dp_block_filtered_keys if k in decoder_params}
        
        # Create one DP block per level.
        dp_blocks_list = []
        for i in range(self.n_levels):
            # Compute spatial size for this level (using the same formula as before)
            spatial_size_level = tuple(s // (2 ** i) for s in self.img_size)
            common_kwargs = {
                "spatial_size": spatial_size_level,
                "in_channels": self.n_input_features_per_level[i] # no concatenation fixed and moving features
            }
            extra_kwargs = {}


            # self.n_features_per_level = decoder_params.get("n_features_per_level", None)
            # self.n_attention_per_level = decoder_params.get("n_attention_per_level", None)

            ### IMPROVEMENT NOTE: Might be better to unify the key names!!!
            # For dp_vfa_std, expect separate lists for attention channels.
            if dp_block_type == "dp_vfa_std":
                extra_kwargs["attn_embed_channels"] = self.n_attention_per_level[i]
            # For dp_vfa_conv, expect separate lists for conv and attention channels.
            elif dp_block_type == "dp_vfa_conv":
                if "n_features_per_level" not in decoder_params or "n_attention_per_level" not in decoder_params:
                    raise ValueError("For dp_vfa_conv, please provide 'n_features_per_level' and 'n_attention_per_level' in decoder_params.")
                extra_kwargs["conv_out_channels"] = self.n_features_per_level[i]
                extra_kwargs["attn_embed_channels"] = self.n_attention_per_level[i]
            # (For future dp block types, extra_kwargs can be set accordingly.)
            
            # Instantiate the block.
            block = dp_block_cls(**common_kwargs, **extra_kwargs, **dp_block_kwargs)
            dp_blocks_list.append(block)
        # Reverse the order so that index 0 is the coarsest level.
        self.dp_blocks = nn.ModuleList(dp_blocks_list[::-1])

    def _upsample_n_times(self, ddf, n_times):
        out = ddf
        for _ in range(n_times):
            out = self.upsample_ddf(out)
        return out

    def forward(self, list_features_1: Sequence[torch.Tensor], 
                      list_features_2: Sequence[torch.Tensor],
                      return_intermediates: bool = False) -> torch.Tensor:
        """
        Both list_features_1 and list_features_2 are lists (length n_levels) of feature maps.
        Each feature map at level i is assumed to have shape:
             [B, n_input_features_per_level[i], H/2^(n_levels-1-i), W/2^(n_levels-1-i), D/2^(n_levels-1-i)]
        
        Processing:
          - Reverse the lists so that index 0 is the coarsest level.
          - At the coarsest level: predict ddf_res[0] from the two features.
          - For levels 1 to n_levels-1:
              * Upsample ddf_tot[i-1].
              * Warp list_features_1[i] using the transformer at level i and the upsampled ddf.
              * Predict ddf_res[i] via DPBlockVFAStandard from the warped feature and list_features_2[i].
              * Compose the upsampled ddf with ddf_res[i] using the composer.
          - Return the final ddf.
        """
        if len(list_features_1) != self.n_levels or len(list_features_2) != self.n_levels:
            raise ValueError(f"Expected both feature lists to have {self.n_levels} elements.")

        # Reverse lists so that index 0 is the coarsest level.
        list_features_1 = list_features_1[::-1]
        list_features_2 = list_features_2[::-1]
        
        ddf_tot = [None] * self.n_levels
        ddf_res = [None] * self.n_levels
        
        # Level 0: coarsest level (no previous ddf).
        # Concatenate features along channel dimension.
        # fused0 = torch.cat([list_features_1[0], list_features_2[0]], dim=1) # ? it seemed to be not used?
        ddf_res[0] = self.dp_blocks[0](list_features_1[0], list_features_2[0])
        ddf_tot[0] = ddf_res[0]
        
        # For levels 1 to n_levels-1 (finer levels)
        for i in range(1, self.n_levels):
            ddf_tot_up = self.upsample_ddf(ddf_tot[i-1])
            # Warp the moving feature at this level using the upsampled ddf.
            warped_feature = self.transformers[i](list_features_1[i], ddf_tot_up)
            # Concatenate warped moving and fixed features along channel dimension.
            # fused = torch.cat([warped_feature, list_features_2[i]], dim=1) # ? it seemed to be not used?
            ddf_res[i] = self.dp_blocks[i](warped_feature, list_features_2[i])
            ddf_tot[i] = self.composers[i](ddf_tot_up, ddf_res[i])
        
        final_ddf = ddf_tot[-1]
        
        if return_intermediates:
            upsampled_ddf_tot = []
            upsampled_ddf_res = []
            for i in range(self.n_levels):
                steps = self.n_levels - 1 - i
                upsampled_tot = self._upsample_n_times(ddf_tot[i], steps) if ddf_tot[i] is not None else None
                upsampled_res = self._upsample_n_times(ddf_res[i], steps) if ddf_res[i] is not None else None
                upsampled_ddf_tot.append(upsampled_tot)
                upsampled_ddf_res.append(upsampled_res)
            return final_ddf, (upsampled_ddf_tot, upsampled_ddf_res)
        else:
            return final_ddf
