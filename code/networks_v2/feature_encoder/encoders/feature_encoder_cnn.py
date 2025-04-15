# feature_encoder/encoders/feature_encoder_cnn.py

import torch
import torch.nn as nn

from ..registries import register_encoder, BLOCK_REGISTRY

# make this global for readability
fe_block_filtered_keys = [
    'kernel_size', 'bias', 
    'n_convolutions', 'res_skip', 
    'act_name', 'norm_name'
]

@register_encoder('cnn_std')
class FeatureEncoderCNN(nn.Module):
    """
    CNN-based multi-resolution encoder with cascaded refinement.
    
    This variant uses:
      - Level 0: A projection operator (1x1 conv) followed by a convolution block without downsampling.
      - Levels 1..(n_levels-1): Each block takes the previous level’s features and performs downsampling.
    
    Parameters:
        n_input_channels (int): Number of input channels.
        encoder_params (dict): Must contain:
            - 'n_input_channels': should match the top-level input channels.
            - 'n_levels': total number of pyramid levels.
            - 'n_features_per_level': list of output feature channels per level.
            - 'block_type': registry key for the convolution block to use (e.g. "convblock_std_stride").
            - 'init_method': initialization method (default "kaiming").
            - (Any additional parameters will be passed to the block.)
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
            
        # Level 0: projection and a conv block (without downsampling)
        self.projection = nn.Conv3d(self.n_input_channels, self.n_features_per_level[0], kernel_size=1, padding=0)
        self.block0 = block_cls(
            in_channels=self.n_features_per_level[0],
            out_channels=self.n_features_per_level[0],
            downsample=False,
            **block_kwargs
        )
        
        # Levels 1...n_levels-1: cascaded blocks with downsampling.
        self.blocks = nn.ModuleList()
        for i in range(1, self.n_levels):
            block = block_cls(
                in_channels=self.n_features_per_level[i-1],
                out_channels=self.n_features_per_level[i],
                downsample=True,
                **block_kwargs
            )
            self.blocks.append(block)
            
        # Initialization (default: kaiming)
        init_method = encoder_params.get('init_method', 'kaiming')
        if init_method.lower() == 'kaiming':
            self.apply(self._init_kaiming)
    
    def _init_kaiming(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=0.2) # leaky relu
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = []
        # Level 0
        x0 = self.projection(x)
        x0 = self.block0(x0)
        features.append(x0)
        # Cascaded processing for subsequent levels
        x_prev = x0
        for block in self.blocks:
            x_prev = block(x_prev)
            features.append(x_prev)
        return features
