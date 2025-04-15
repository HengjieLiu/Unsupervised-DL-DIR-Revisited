# feature_encoder/encoders/feature_encoder_cnn_pooled.py

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from ..registries import register_encoder, BLOCK_REGISTRY

# make this global for readability
fe_block_filtered_keys = [
    'kernel_size', 'bias', 
    'n_convolutions', 'res_skip', 
    'act_name', 'norm_name'
]

@register_encoder('cnn_pooled')
class FeatureEncoderCNN_PooledInput(nn.Module):
    """
    CNN-based multi-resolution encoder that uses average pooling on the input
    to generate different scales. For each level, a projection (1x1 conv) and a conv block are applied.
    
    Parameters:
        n_input_channels (int): Number of input channels.
        encoder_params (dict): Must contain:
            - 'n_input_channels': should match the top-level input channels.
            - 'n_levels': total number of pyramid levels.
            - 'n_features_per_level': list of output feature channels per level.
            - 'block_type': registry key for the convolution block to use (e.g. "convblock_std_stride").
            - 'init_method': initialization method (default "kaiming").
            - (Additional parameters are passed to the block.)
    """
    def __init__(self, **encoder_params):
        super().__init__()
        
        # self.img_size = decoder_params["img_size"] # not used
        self.n_input_channels = encoder_params["n_input_channels"]
        self.n_levels = encoder_params["n_levels"]
        self.n_features_per_level = encoder_params["n_features_per_level"]
        
        # Check consistency
        if self.n_levels != len(self.n_features_per_level):
            raise ValueError("n_levels must equal the length of n_features_per_level.")
            
        # Lookup the convolution block type from BLOCK_REGISTRY.
        block_type_key = encoder_params.get('block_type')
        block_cls = BLOCK_REGISTRY.get(block_type_key)
        if block_cls is None:
            raise ValueError(f"Unsupported block type: {block_type_key}")

        # Filter keys for block_cls
        block_kwargs = {k: encoder_params[k] for k in fe_block_filtered_keys if k in encoder_params}
        # print('encoder_params: ', encoder_params)
        # print('block_kwargs: ', block_kwargs)
        # block_kwargs = encoder_params
            
        # For each level, create a projection and a conv block.
        self.projections = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(self.n_levels):
            self.projections.append(nn.Conv3d(self.n_input_channels, self.n_features_per_level[i], kernel_size=1, padding=0))
            self.blocks.append(block_cls(
                in_channels=self.n_features_per_level[i],
                out_channels=self.n_features_per_level[i],
                downsample=False,
                **block_kwargs
            ))
            
        # Initialization (default: kaiming)
        self.init_method = encoder_params.get('init_method', 'kaiming')
        if self.init_method.lower() == 'kaiming':
            self.apply(self._init_kaiming)
    
    def _init_kaiming(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=0.2) # leaky relu
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = []
        x_current = x
        # For each level, downsample the input progressively using average pooling.
        for i, (proj, block) in enumerate(zip(self.projections, self.blocks)):
            if i > 0:
                x_current = nnf.avg_pool3d(x_current, kernel_size=2, stride=2)
            x_proj = proj(x_current)
            feat = block(x_proj)
            features.append(feat)
        return features
