# feature_encoder/feature_encoder.py

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

### import registry
from .registries import FEATURE_ENCODER_REGISTRY

### import shape_logic from SITReg
import sys
dir_shape_logic = '/homebase/DL_projects/wavereg/SITReg/src/algorithm'
sys.path.insert(0, dir_shape_logic)
from shape_logic import EncoderShapeLogic


class FeatureEncoder(nn.Module):

    """
    Top-level feature encoder that instantiates a specific encoder type.

    Arguments:
        encoder_type: the type of encoder to instantiate.
        encoder_params: dictionary of parameters needed by the chosen encoder.
        Optional arguments:
            !!! Note: these arguments are expected in encoder_params and will be overwritten by encoder_params if mismatch
            img_size: [H, W, D]
            n_input_channels: number of input channels (example: 1).
            n_levels: total number of pyramid levels.
                if n_levels = 5, there will be 4 downsampling
            n_features_per_level: number of channels at each level.

    Forward method:
        Takes tensor of shape [B, n_input_channels, H, W, D]
        Returns a list of tensors of shapes:
            [B, n_features_per_level[i], H/2^i, W/2^i, D/2^i] for i=0,...,n_levels-1.

    List of methods:
        __init__
        _check_parameters
        _get_downsampling_factors
        forward
        freeze
        unfreeze
        get_shapes
        
    The following attributes and methods/functions are adapted from SITReg's feature_extractor:
    (Ref: https://github.com/honkamj/SITReg/blob/main/src/model/sitreg/feature_extractor.py)    
        self.shape_logic
        get_shapes
        _get_downsampling_factors
    """
    
    def __init__(
        self, 
        encoder_type: str,
        encoder_params: dict,
        img_size: Optional[Sequence[int]] = None,
        n_input_channels: Optional[int] = None,
        n_levels: Optional[int] = None,
        n_features_per_level: Optional[Sequence[int]] = None,
        CHECK: bool = True,
    ) -> None:
        
        super().__init__()

        ### Retrieve parameters and assign
        # Helper function to retrieve a parameter from encoder_params.
        # If a value is provided to the initializer, it checks for consistency.
        def get_param(param_name, provided_value):
            if provided_value is None:
                return encoder_params[param_name]
            else:
                if encoder_params[param_name] != provided_value:
                    print(
                        f"Warning: Provided {param_name} ({provided_value}) does not match "
                        f"encoder_params value ({encoder_params[param_name]}). Using encoder_params value."
                    )
                return encoder_params[param_name]
        
        # Retrieve key parameters from encoder_params, with optional consistency check
        self.img_size = get_param("img_size", img_size)
        self.n_input_channels = get_param("n_input_channels", n_input_channels)
        self.n_levels = get_param("n_levels", n_levels)
        self.n_features_per_level = get_param("n_features_per_level", n_features_per_level)

        
        ### Call sanity check method
        if CHECK:
            self._check_parameters()
        

        ### Instantiate the encoder using encoder_params.
        # Lookup the encoder class from the registry
        encoder_cls = FEATURE_ENCODER_REGISTRY.get(encoder_type.lower())
        if encoder_cls is None:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        self.encoder = encoder_cls(**encoder_params)


        ### Set up shape_logic 
        """from SITReg's feature_extractor"""
        self.shape_logic = EncoderShapeLogic(
            shape_mode="ceil",
            n_feature_levels=self.n_levels, # previous: len(n_features_per_resolution)
            input_shape=self.img_size,
            downsampling_factor=2,
        )
        

    def _check_parameters(self):
        """
        Sanity check for encoder parameters to ensure consistency and correctness.
        
        Raises:
            ValueError: If any parameter is inconsistent or invalid.
        """
        # Check that n_levels matches the length of n_features_per_level.
        if self.n_levels != len(self.n_features_per_level):
            raise ValueError(
                f"Mismatch in FeatureEncoder: n_levels ({self.n_levels}) must be equal to "
                f"len(n_features_per_level) ({len(self.n_features_per_level)}).\n"
                f"Provided n_features_per_level: {self.n_features_per_level}"
            )
    
        # Additional checks can be added here as needed.
        # ...


    def forward(self, x: Tensor) -> Sequence[Tensor]:
        list_features = self.encoder(x)
        return list_features

    def freeze(self):
        """Freeze the weights (set requires_grad=False)."""
        print('#'*60)
        print('Freeze the FeatureEncoder weights')
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze the weights (set requires_grad=True)."""
        print('#'*60)
        print('Unfreeze the FeatureEncoder weights')
        for param in self.parameters():
            param.requires_grad = True

    """from SITReg's feature_extractor"""
    def get_shapes(self) -> Sequence[Sequence[int]]:
        # from SITReg's feature_extractor
        return [
            [n_features] + list(volume_shape)
            for n_features, volume_shape in zip(
                self.n_features_per_level, self.shape_logic.calculate_shapes()
            ) # self._n_features_per_resolution, self.shape_logic.calculate_shapes()
        ]
        
    """from SITReg's feature_extractor"""
    def _get_downsampling_factors(self) -> Sequence[Sequence[float]]:
        return self.shape_logic.calculate_downsampling_factors()
    
    
