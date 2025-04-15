# deformation_decoder/deformation_decoder.py

from typing import Optional, Sequence
from typing import overload

import torch
import torch.nn as nn
from torch import Tensor

### import registry
from .registries import DEFORMATION_DECODER_REGISTRY


class DeformationDecoder(nn.Module):
    """
    Top-level deformation decoder that instantiates a specific decoder type.

    Arguments:
        decoder_type: the type of decoder to instantiate.
        decoder_params: dictionary of parameters needed by the chosen decoder.
        Optional arguments:
            !!! Note: these arguments are expected in decoder_params and will be overwritten by decoder_params if mismatch
            img_size: [H, W, D]
            n_output_channels: number of channels in the output deformation field (example: 3).
            n_levels: total number of pyramid levels.
                if n_levels = 5, there will be 4 downsampling
            n_input_features_per_level: list of number of input feature channels at each level.
                ??? # number of features in list_features_1/list_features_2
            n_features_per_level: list of number of decoder channels at each level (for further processing).
                level 0 to n_levels-1, will NOT be reversed later (e.g. in DeformationDecoderPyramidalCNN)
                ??? # control channel size of decoder (level 0 to n_levels-1, will NOT be reversed later)

    List of methods:
        __init__
        _check_parameters
        _check_inputs
        forward
        freeze
        unfreeze
        
    Forward method:
        Takes two list of feature maps (length n_levels), where each feature map has shape
            [B, n_input_features_per_level[i], H/2^i, W/2^i, D/2^i] for i=0,...,n_levels-1.
        Returns a deformation field of shape [B, n_output_channels, H, W, D].
        Optionally, it can output a list of intermediate deformation fields.
    """
    def __init__(
        self,
        decoder_type: str, 
        decoder_params: dict,
        img_size: Optional[Sequence[int]] = None,
        n_output_channels: Optional[int] = None,
        n_levels: Optional[int] = None,
        n_input_features_per_level: Optional[Sequence[int]] = None,
        n_features_per_level: Optional[Sequence[int]] = None, 
        n_attention_per_level: Optional[Sequence[int]] = None, 
        CHECK: bool = True,
    ) -> None:
        
        super().__init__()

        ### Retrieve parameters and assign
        # Helper function to retrieve a parameter from decoder_params.
        # If a value is provided to the initializer, it checks for consistency.
        def get_param(param_name, provided_value):
            if provided_value is None:
                if param_name not in decoder_params:
                    return None
                else:
                    return decoder_params[param_name]
            else:
                if decoder_params[param_name] != provided_value:
                    print(
                        f"Warning: Provided {param_name} ({provided_value}) does not match "
                        f"decoder_params value ({decoder_params[param_name]}). Using decoder_params value."
                    )
                return decoder_params[param_name]
        
        # Retrieve key parameters from decoder_params, with optional consistency check
        self.img_size = get_param("img_size", img_size)
        self.n_output_channels = get_param("n_output_channels", n_output_channels)
        self.n_levels = get_param("n_levels", n_levels)
        self.n_input_features_per_level = get_param("n_input_features_per_level", n_input_features_per_level)
        self.n_features_per_level = get_param("n_features_per_level", n_features_per_level)
        self.n_attention_per_level = get_param("n_attention_per_level", n_attention_per_level)
        
        ### Call sanity check method
        if CHECK:
            self._check_parameters()

        
        ### Instantiate the decoder using decoder_params.
        # Lookup the specific decoder from the registry.
        decoder_cls = DEFORMATION_DECODER_REGISTRY.get(decoder_type.lower())
        if decoder_cls is None:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
        self.decoder = decoder_cls(**decoder_params)

    
    def _check_parameters(self):
        """
        Sanity check for encoder parameters to ensure consistency and correctness.
        
        Raises:
            ValueError: If any parameter is inconsistent or invalid.
        """
        # Check that n_levels matches the length of n_features_per_level.
        if self.n_input_features_per_level is not None and self.n_levels != len(self.n_input_features_per_level):
            raise ValueError(
                f"Mismatch in DeformationDecoder: n_levels ({self.n_levels}) must be equal to "
                f"len(n_input_features_per_level) ({len(self.n_input_features_per_level)}).\n"
                f"Provided n_input_features_per_level: {self.n_input_features_per_level}"
            )
        if self.n_features_per_level is not None and self.n_levels != len(self.n_features_per_level):
            raise ValueError(
                f"Mismatch in DeformationDecoder: n_levels ({self.n_levels}) must be equal to "
                f"len(n_features_per_level) ({len(self.n_features_per_level)}).\n"
                f"Provided n_features_per_level: {self.n_features_per_level}"
            )
        if self.n_attention_per_level is not None and self.n_levels != len(self.n_attention_per_level):
            raise ValueError(
                f"Mismatch in DeformationDecoder: n_levels ({self.n_levels}) must be equal to "
                f"len(n_attention_per_level) ({len(self.n_attention_per_level)}).\n"
                f"Provided n_attention_per_level: {self.n_attention_per_level}"
            )
    
        # Additional checks can be added here as needed.
        # ...

    def _check_inputs(self, list_features_1, list_features_2):
        """
        Sanity check for inputs in the forward method to ensure consistency and correctness.
        
        Raises:
            ValueError: If any parameter is inconsistent or invalid.
        """
        # 1. Check that both lists have length equal to n_levels.
        if len(list_features_1) != self.n_levels:
            raise ValueError(f"Expected list_features_1 to have {self.n_levels} elements, got {len(list_features_1)}")
        if len(list_features_2) != self.n_levels:
            raise ValueError(f"Expected list_features_2 to have {self.n_levels} elements, got {len(list_features_2)}")
        
        # 2. For each level, check the shape of feature maps in list_features_1.
        # Expected shape: [B, n_input_features_per_level[i], H//(2**i), W//(2**i), D//(2**i)]
        for i in range(self.n_levels):
            feat_shape_1 = list_features_1[i].shape
            feat_shape_2 = list_features_2[i].shape
            expected_channels = self.n_input_features_per_level[i]
            expected_spatial = tuple(self.img_size[j] // (2 ** i) for j in range(len(self.img_size)))
            if feat_shape_1[1] != expected_channels:
                raise ValueError(
                    f"Level {i}: Expected {expected_channels} channels in list_features_1, got {feat_shape_1[1]}"
                )
            if feat_shape_1[2:] != expected_spatial:
                raise ValueError(
                    f"Level {i}: Expected spatial dimensions {expected_spatial} in list_features_1, got {feat_shape_1[2:]}"
                )
            if feat_shape_2[1] != expected_channels:
                raise ValueError(
                    f"Level {i}: Expected {expected_channels} channels in list_features_1, got {feat_shape_2[1]}"
                )
            if feat_shape_2[2:] != expected_spatial:
                raise ValueError(
                    f"Level {i}: Expected spatial dimensions {expected_spatial} in list_features_1, got {feat_shape_2[2:]}"
                )

    @overload
    def forward(
        self, 
        list_features_1: Sequence[Tensor], 
        list_features_2: Sequence[Tensor], 
        return_intermediates: bool = False
    ) -> Tensor:
        ...
    
    @overload
    def forward(
        self, 
        combined_features: Sequence[Tensor], 
        return_intermediates: bool = False
    ) -> Tensor:
        ...

    def forward(
        self, 
        *args, 
        return_intermediates: bool = False,
        CHECK: bool = False,
    ) -> Tensor:
        """
        Args:
            For the two-list case:
              list_features_1, list_features_2: lists of feature maps, each of length n_levels.
                Each feature map at level i is expected to have shape:
                [B, n_input_features_per_level[i], H/2^i, W/2^i, D/2^i]
            For the combined case:
              combined_features: list of feature maps with batch dimension equal to sum of two batches.
            return_intermediates: if True, returns (deformation_field, intermediates)
            CHECK: flag to enable runtime input checks

        Returns:
            deformation_field: tensor of shape [B, n_output_channels, H, W, D]
            (Optionally) a list of intermediate deformation fields.
        """

        if len(args) == 1:
            ### input
            combined_features = args[0]

            ### forward
            outputs = self.decoder(combined_features, return_intermediates)
        elif len(args) == 2:
            ### input
            list_features_1, list_features_2 = args
            
            ### check
            if CHECK:
                self._check_inputs()
    
            ### forward
            outputs = self.decoder(list_features_1, list_features_2, return_intermediates)
        else:
            raise ValueError("Invalid number of arguments. Expected either one combined feature list or two separate feature lists.")
            
        ### return
        if isinstance(outputs, tuple):
            if return_intermediates:
                return outputs
            else:
                return outputs[0]
        else:
            return outputs


    def freeze(self):
        """Freeze the weights (set requires_grad=False)."""
        print('#'*60)
        print('Freeze the DeformationDecoder weights')
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze the weights (set requires_grad=True)."""
        print('#'*60)
        print('Unfreeze the DeformationDecoder weights')
        for param in self.parameters():
            param.requires_grad = True


    
