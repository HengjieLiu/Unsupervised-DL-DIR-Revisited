# deformation_decoder/decoders/deformation_decoder_pyramidal_cnn.py

import torch
import torch.nn as nn
import torch.nn.init as init
        
from typing import Sequence

from ..registries import register_deformation_decoder

from ..building_blocks.deformation_prediction_block_conv_block_standard import DPBlockConvBlockStandard
from ..building_blocks.block_utils import FlowConv  # ensure FlowConv is imported


import sys
sys.path.insert(0, '/homebase/DL_projects/wavereg/code')
import utils_warp  # Assumed to provide SpatialTransformer, ComposeDVF, dvf_upsample
import utils_correlation

# make this global for readability
dp_block_filtered_keys = [
    'kernel_size', 'bias', 
    'n_convolutions', 'res_skip', 
    'act_name', 'norm_name', 
    'use_flowconv', 'flowconv_type',
]

@register_deformation_decoder("pyramidal_cnn_std")
class DeformationDecoderPyramidalCNN(nn.Module):
    """
    Deformation decoder based on a pyramidal CNN.
    
    This decoder takes two lists of multi-scale features (from moving and fixed images)
    arranged from fine (level 0) to coarse (level n_levels-1). It reverses the order so
    that processing starts at the coarsest level and proceeds from coarse to fine.
    
    At the coarsest level, a deformation prediction block (DP block) predicts a residual DDF.
    For each finer level, the previous DDF is upsampled and used to warp the moving features,
    then a new residual DDF is predicted and composed with the upsampled DDF.
    
    The final output is the DDF at the finest resolution.
    
    All required parameters (img_size, n_output_channels, n_levels, n_input_features_per_level, 
    n_features_per_level, and DP block parameters) are provided via decoder_params.
    """
    def __init__(self, **decoder_params):
        super().__init__()
        
        # Extract parameters from decoder_params
        self.img_size = decoder_params["img_size"]  # e.g., (H, W, D)
        self.n_output_channels = decoder_params["n_output_channels"]
        self.n_levels = decoder_params["n_levels"]
        self.n_input_features_per_level = decoder_params["n_input_features_per_level"]
        self.n_features_per_level = decoder_params["n_features_per_level"] # do not reverse!!! (This is correct)
        # self.n_features_per_level = decoder_params["n_features_per_level"][::-1] # reverse!
        self.add_feat = decoder_params["add_feat"]
        print('self.n_input_features_per_level: ', self.n_input_features_per_level)
        print('self.n_features_per_level: ', self.n_features_per_level)
        print('self.add_feat: ', self.add_feat)


        # self.add_feat 
        # null
        # {"name": "diff"}
        # {"name": "diffonly"}
        # {"name": "corr", "type": "org/vfa"}
        # {"name": "corronly", "type": "org/vfa"}
        ### potential more:
        # {"name": "diffonly"} # bidirection, sum_diff

        # if self.add_feat is None:
        #     in_channels  = self.n_input_features_per_level[i] * 2
        #     out_channels = self.n_features_per_level[i]
        # elif self.add_feat['name'] == 'diff':
        #     in_channels  = self.n_input_features_per_level[i] * 3
        #     out_channels = self.n_features_per_level[i]
        # elif self.add_feat['name'] == 'diffonly':
        #     in_channels  = self.n_input_features_per_level[i] * 1
        #     out_channels = self.n_features_per_level[i]
        # elif self.add_feat['name'] == 'corr':
        #     in_channels  = self.n_input_features_per_level[i] * 2 + 27
        #     out_channels = self.n_features_per_level[i]
        # elif self.add_feat['name'] == 'corronly':
        #     in_channels  = 27
        #     out_channels = self.n_features_per_level[i]
        # else:
        #     raise ValueError(f"self.add_feat: {self.add_feat} is not recognized")
            

        # Upsampling function for DDFs.
        # self.upsample_ddf = utils_warp.dvf_upsample
        self.upsample_ddf = utils_warp.ResizeFlow(spatial_scale=2, flow_scale=2, ndim=3)

        # Initialize transformers and composers in coarse-to-fine order.
        # For level 0 (coarsest): spatial size = img_size // 2^(n_levels-1)
        self.transformers = nn.ModuleList()
        self.composers = nn.ModuleList()
        for i in range(self.n_levels):
            # Compute spatial size: level i corresponds to 2^(n_levels-1 - i) downsampling.
            spatial_size = tuple(s // (2 ** (self.n_levels - 1 - i)) for s in self.img_size)
            self.transformers.append(utils_warp.SpatialTransformer(spatial_size))
            self.composers.append(utils_warp.ComposeDVF(spatial_size))

        # Filter keys for DP block parameters.
        dp_block_kwargs = {k: decoder_params[k] for k in dp_block_filtered_keys if k in decoder_params}
        
        # Create a deformation prediction block (DP block) for each level.
        # For level i, in_channels = n_input_features_per_level[i]*2 (due to concatenation), and out_channels = n_features_per_level[i].
        # self.dp_blocks = nn.ModuleList([
        #     DPBlockConvBlockStandard(
        #         in_channels=in_channels,
        #         out_channels=out_channels,
        #         **dp_block_kwargs
        #     )
        #     for i in range(self.n_levels)[::-1] # reverse so that we call from lv_n-1 -> ... -> lv_1 -> lv_0
        # ])

        self.dp_blocks = nn.ModuleList()
        for i in range(self.n_levels)[::-1]:  # Reverse: from level n-1 to level 0
            if self.add_feat is None or self.add_feat['name'] == 'diffsum':
                in_channels = self.n_input_features_per_level[i] * 2
            elif self.add_feat['name'] == 'diff':
                in_channels = self.n_input_features_per_level[i] * 3
            elif self.add_feat['name'] == 'diffonly':
                in_channels = self.n_input_features_per_level[i] * 1
            elif self.add_feat['name'] == 'corr':
                in_channels = self.n_input_features_per_level[i] * 2 + 27
            elif self.add_feat['name'] == 'corronly':
                in_channels = 27
            else:
                raise ValueError(f"self.add_feat: {self.add_feat} is not recognized")
        
            out_channels = self.n_features_per_level[i]
            
            self.dp_blocks.append(
                DPBlockConvBlockStandard(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **dp_block_kwargs
                )
            )


        # Initialization (default: kaiming + smallflow)
        self.init_method = decoder_params.get('init_method', 'kaiming')
        self.init_method_flow = decoder_params.get('init_method', 'smallflow')
        if self.init_method.lower() == 'kaiming' and self.init_method_flow.lower() == "smallflow":
            self.apply(self._initialize_weights)
        
    def _initialize_weights(self, module):
        # This method is called by self.apply() for each submodule.
        if isinstance(module, FlowConv):
            if self.init_method_flow.lower() == "smallflow":
                # FlowConv is a nn.Sequential wrapping a Conv3d.
                for m in module:
                    if isinstance(m, nn.Conv3d):
                        init.normal_(m.weight, mean=0, std=1e-5)
                        if m.bias is not None:
                            init.constant_(m.bias, 0)
        elif isinstance(module, nn.Conv3d):
            if self.init_method.lower() == "kaiming":
                init.kaiming_normal_(module.weight, a=0.2)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

        # print("smallflow only") # not showing up in log or jupyter notebook ...
        # if isinstance(module, FlowConv):
        #     if self.init_method_flow.lower() == "smallflow":
        #         # FlowConv is a nn.Sequential wrapping a Conv3d.
        #         for m in module:
        #             if isinstance(m, nn.Conv3d):
        #                 init.normal_(m.weight, mean=0, std=1e-5)
        #                 if m.bias is not None:
        #                     init.constant_(m.bias, 0)
                            
        # print("kaiming only") # not showing up in log or jupyter notebook ...
        # if isinstance(module, nn.Conv3d):
        #     if self.init_method.lower() == "kaiming":
        #         init.kaiming_normal_(module.weight, a=0.2)
        #         if module.bias is not None:
        #             init.constant_(module.bias, 0)
    
    def _upsample_n_times(self, ddf, n_times):
        out = ddf
        for _ in range(n_times):
            out = self.upsample_ddf(out)
        return out

    def _get_input_features(self, features_1, features_2):
        ### by default the order is: moving, fixed
        ### in corr_func the order should be reversed

        if self.add_feat is not None and self.add_feat['name'].startswith('corr'):
            radius = self.add_feat['radius'] 
            if self.add_feat['type'] == 'org':
                corr_func = utils_correlation.WinCorrTorch(radius=radius)
            elif self.add_feat['type'] == 'vfa':
                corr_func = utils_correlation.WinCorrTorch_VFA(radius=radius)
            else:
                raise ValueError(f"self.add_feat: {self.add_feat} is not recognized")

        if self.add_feat is None:
            input_ = torch.cat([features_1, features_2], dim=1)
        elif self.add_feat['name'] == 'diffsum':
            input_ = torch.cat([features_1 - features_2, features_1 + features_2], dim=1)
        elif self.add_feat['name'] == 'diff':
            diff = features_1 - features_2
            # input_ = torch.cat([features_1, features_2, diff], dim=1)
            input_ = torch.cat([features_1, diff, features_2], dim=1)
        elif self.add_feat['name'] == 'diffonly':
            input_ = features_1 - features_2
        elif self.add_feat['name'] == 'corr':
            if not self.add_feat['flip']:
                ### the order of corr should take fixed, moving, note the reversed order here!!!
                corr = corr_func(features_2, features_1)
            else:
                ### for testing!!!
                corr = corr_func(features_1, features_2)

            # input_ = torch.cat([features_1, features_2, corr], dim=1)
            input_ = torch.cat([features_1, corr, features_2], dim=1)
        elif self.add_feat['name'] == 'corronly':
            if not self.add_feat['flip']:
                ### the order of corr should take fixed, moving, note the reversed order here!!!
                input_ = corr_func(features_2, features_1)
            else:
                ### for testing!!!
                input_ = corr_func(features_1, features_2)
        else:
            raise ValueError(f"self.add_feat: {self.add_feat} is not recognized")

        return input_
    
    def forward(
        self, 
        list_features_1: Sequence[torch.Tensor], 
        list_features_2: Sequence[torch.Tensor],
        return_intermediates: bool = False
    ) -> torch.Tensor:

        """
        Args:
            list_features_1, list_features_2: Lists of feature maps (length = n_levels), where each feature map at level i
              is expected to have shape [B, n_input_features_per_level[i], H//(2^(n_levels-1-i)), W//(2^(n_levels-1-i)), D//(2^(n_levels-1-i))].
            return_intermediates: if True, returns (final_dvf, (ddf_tot_upsampled, ddf_res_upsampled))
        
        Processing:
            - At the coarsest level (i = 0): predict ddf_res[0] from list_features_1[0] and list_features_2[0]; set ddf_tot[0] = ddf_res[0].
            - For levels i = 1 to n_levels-1:
                * Upsample ddf_tot[i-1].
                * Warp list_features_1[i] using the transformer at level i and the upsampled DVF.
                * Predict ddf_res[i] using DP[i] from the warped features and list_features_2[i].
                * Compose the upsampled DDF with ddf_res[i] using the composer at level i to obtain ddf_tot[i].
            - The final output DDF is ddf_tot[n_levels-1].
        """
        if len(list_features_1) != self.n_levels or len(list_features_2) != self.n_levels:
            raise ValueError(f"Expected both feature lists to have {self.n_levels} elements.")

        # Reverse the lists so that index 0 is the coarsest level.
        list_features_1 = list_features_1[::-1]
        list_features_2 = list_features_2[::-1]

        ddf_tot = [None] * self.n_levels
        ddf_res = [None] * self.n_levels

        for i in range(self.n_levels):
            if i == 0:
                # For level 0, no upsampling or transformation is needed.
                input_ = self._get_input_features(list_features_1[i], list_features_2[i])
                ddf_res[i] = self.dp_blocks[i](input_)
                ddf_tot[i] = ddf_res[i]
            else:
                # For finer levels, use the previous result.
                ddf_tot_up = self.upsample_ddf(ddf_tot[i - 1])
                warped_list_features_1 = self.transformers[i](list_features_1[i], ddf_tot_up)
                input_ = self._get_input_features(warped_list_features_1, list_features_2[i])
                ddf_res[i] = self.dp_blocks[i](input_)
                ddf_tot[i] = self.composers[i](ddf_tot_up, ddf_res[i])
        
        # # Process from coarse (level 0) to fine (level n_levels-1)
        # # Level 0 (coarsest): no previous DDF.
        # ddf_res[0] = self.dp_blocks[0](list_features_1[0], list_features_2[0])
        # ddf_tot[0] = ddf_res[0]
        
        # # For levels 1 to n_levels-1 (finer levels)
        # for i in range(1, self.n_levels):
        #     ddf_tot_up = self.upsample_ddf(ddf_tot[i-1])
        #     warped_list_features_1 = self.transformers[i](list_features_1[i], ddf_tot_up)
        #     ddf_res[i] = self.dp_blocks[i](warped_list_features_1, list_features_2[i])
        #     ddf_tot[i] = self.composers[i](ddf_tot_up, ddf_res[i])
        
        final_ddf = ddf_tot[-1]  # Finest resolution DDF
        
        if return_intermediates:
            # Upsample each intermediate DDF to the finest resolution.
            upsampled_ddf_tot = []
            upsampled_ddf_res = []
            for i in range(self.n_levels):
                # Number of upsampling steps: (n_levels-1 - i)
                steps = self.n_levels - 1 - i
                upsampled_tot = self._upsample_n_times(ddf_tot[i], steps) if ddf_tot[i] is not None else None
                upsampled_res = self._upsample_n_times(ddf_res[i], steps) if ddf_res[i] is not None else None
                upsampled_ddf_tot.append(upsampled_tot)
                upsampled_ddf_res.append(upsampled_res)
            return final_ddf, (upsampled_ddf_tot, upsampled_ddf_res)
        else:
            return final_ddf
        