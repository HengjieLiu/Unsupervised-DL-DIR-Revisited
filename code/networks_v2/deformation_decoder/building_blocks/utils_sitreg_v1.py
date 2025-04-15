

"""
List of classes:
    from https://github.com/honkamj/SITReg/blob/main/src/model/sitreg/model.py
        MappingPair
        _MappingBuilder
        _BaseTransformationExtractionNetwork_my_v1
        _DenseExtractionNetwork_my_v1
        
"""

### from SITReg
# ref: https://github.com/honkamj/SITReg/blob/main/src/model/sitreg/model.py
from abc import abstractmethod
# from itertools import count
from logging import getLogger
from typing import NamedTuple, Sequence
# from typing import NamedTuple, Optional, Sequence, cast

### composable_mapping
from composable_mapping import (
    CoordinateSystem,
    CubicSplineSampler,
    DataFormat,
    GridComposableMapping,
    Identity,
    LinearInterpolator,
    OriginalFOV,
    Start,
    affine,
    default_sampler,
    samplable_volume,
)

### deformation_inversion_layer
from deformation_inversion_layer.interface import FixedPointSolver

from numpy import prod as np_prod
import torch
from torch import Tensor
from torch import device as torch_device
from torch import float64, long, tanh
from torch.nn import Linear, Module, ModuleList
import torch.nn as nn


### my building blocks
from ..registries import register_deformation_decoder_block
from .block_utils import get_activation, get_normalization

# from .block_utils import FlowConv
from ..building_blocks.flowconv_family import VanillaFlowConv, PerChannelFlowConv, SEFlowConv, CBAMFlowConv


import sys
sys.path.insert(0, '/homebase/DL_projects/wavereg/code')
# import utils_warp  # Assumed to provide SpatialTransformer, ComposeDVF, dvf_upsample
import utils_correlation
corr_func_org = utils_correlation.WinCorrTorch(radius=1)
corr_func_vfa = utils_correlation.WinCorrTorch_VFA(radius=1)



logger = getLogger(__name__)



class MappingPair(NamedTuple):
    """Mapping pair containing both forward and inverse deformation"""

    forward_mapping: GridComposableMapping
    inverse_mapping: GridComposableMapping


class _MappingBuilder:
    """Builder peforming the anti-symmetric deformation updates"""

    def __init__(
        self,
        forward_affine: GridComposableMapping,
        inverse_affine: GridComposableMapping,
        resample_when_composing: bool,
    ) -> None:
        self._resample_when_composing = resample_when_composing
        self.forward_affine = forward_affine
        self.inverse_affine = inverse_affine
        self.left_forward_dense = Identity(
            device=forward_affine.device, dtype=forward_affine.dtype
        ).assign_coordinates(forward_affine)
        self.right_forward_dense = Identity(
            device=forward_affine.device, dtype=forward_affine.dtype
        ).assign_coordinates(forward_affine)
        self.left_inverse_dense = Identity(
            device=forward_affine.device, dtype=forward_affine.dtype
        ).assign_coordinates(forward_affine)
        self.right_inverse_dense = Identity(
            device=forward_affine.device, dtype=forward_affine.dtype
        ).assign_coordinates(forward_affine)

    def left_forward(self) -> GridComposableMapping:
        """Return full left forward mapping"""
        return self.forward_affine @ self.left_forward_dense

    def right_forward(self) -> GridComposableMapping:
        """Return full right forward mapping"""
        return self.inverse_affine @ self.right_forward_dense

    def left_inverse(self) -> GridComposableMapping:
        """Return full left inverse mapping"""
        return self.left_inverse_dense @ self.inverse_affine

    def right_inverse(self) -> GridComposableMapping:
        """Return full right inverse mapping"""
        return self.right_inverse_dense @ self.forward_affine

    def _resample(
        self,
        mapping: GridComposableMapping,
    ) -> GridComposableMapping:
        if self._resample_when_composing:
            return mapping.resample()
        return mapping

    def update(
        self,
        forward_dense: GridComposableMapping,
        inverse_dense: GridComposableMapping,
    ) -> None:
        """Update with mappings from new stage"""
        self.left_forward_dense = self._resample(
            self.left_forward_dense @ forward_dense
        )
        self.right_forward_dense = self._resample(
            self.right_forward_dense @ inverse_dense
        )
        self.left_inverse_dense = self._resample(
            inverse_dense @ self.left_inverse_dense
        )
        self.right_inverse_dense = self._resample(
            forward_dense @ self.right_inverse_dense
        )

    def as_mapping_pair(self, include_affine: bool = True) -> MappingPair:
        """Get current mapping as mapping pair"""
        if include_affine:
            forward = self._resample(
                self.left_forward() @ self.right_inverse(),
            )
            inverse = self._resample(
                self.right_forward() @ self.left_inverse(),
            )
        else:
            forward = self._resample(
                self.left_forward_dense @ self.right_inverse_dense,
            )
            inverse = self._resample(self.right_forward_dense @ self.left_inverse_dense)
        return MappingPair(forward, inverse)


class _BaseTransformationExtractionNetwork_my_v1(Module):
    """
    My NOTE:
        The 2 directions happened in this function: network(image_1, image_2) vs network(image_2, image_1)
    Key implementation:
        _extract_atomic_transformations calls _extract_atomic_transformation in 2 directions
    Key question:
        why do the difference and summation?
            (input_1_modified - input_2_modified, input_1_modified + input_2_modified),
            (input_2_modified - input_1_modified, input_1_modified + input_2_modified),
        
    """
    
    """Base class for generating exactly inverse consistent transformations"""

    @abstractmethod
    def _extract_atomic_transformation(
        self,
        combined_input: Tensor,
    ) -> GridComposableMapping:
        """Extract the smallest unit transformation"""

    @abstractmethod
    def _invert_mapping(
        self, mapping: GridComposableMapping, device: torch_device
    ) -> GridComposableMapping:
        """Invert transformation"""

    def _modify_input(self, input_tensor: Tensor) -> Tensor:
        return input_tensor

    def _extract_atomic_transformations(
        self,
        features_1: Tensor,
        features_2: Tensor,
        add_feat = None, # added by Hengjie
        # sum_diff: bool = True, # added by Hengjie
    ) -> tuple[GridComposableMapping, GridComposableMapping]:
        input_1_modified = self._modify_input(features_1)
        input_2_modified = self._modify_input(features_2)

        ### original
        # forward_combined_input = torch.cat((input_1_modified - input_2_modified, input_1_modified + input_2_modified), dim=1,)
        # reverse_combined_input = torch.cat((input_2_modified - input_1_modified, input_1_modified + input_2_modified), dim=1,)

        ### HJ DEBUG
        # print(f"$$$$$$$$$$$$$ in _BaseTransformationExtractionNetwork_my_v1: _extract_atomic_transformations sum_diff {sum_diff}")
        # if sum_diff: # added by Hengjie
        #     forward_combined_input = torch.cat((input_1_modified - input_2_modified, input_1_modified + input_2_modified), dim=1,)
        #     reverse_combined_input = torch.cat((input_2_modified - input_1_modified, input_1_modified + input_2_modified), dim=1,)
        # else: # added by Hengjie
        #     forward_combined_input = torch.cat((input_1_modified, input_2_modified), dim=1,)
        #     reverse_combined_input = torch.cat((input_2_modified, input_1_modified), dim=1,)

        ### HJ DEBUG
        # print(f"$$$$$$$$$$$$$ in _BaseTransformationExtractionNetwork_my_v1: _extract_atomic_transformations add_feat {add_feat}")
        if add_feat is None:
            forward_combined_input = torch.cat((input_1_modified, input_2_modified), dim=1,)
            reverse_combined_input = torch.cat((input_2_modified, input_1_modified), dim=1,)
        elif add_feat['name'] == 'diffsum':
            forward_combined_input = torch.cat((input_1_modified - input_2_modified, input_1_modified + input_2_modified), dim=1,)
            reverse_combined_input = torch.cat((input_2_modified - input_1_modified, input_1_modified + input_2_modified), dim=1,)
        elif add_feat['name'] == 'corronly':
            ### need to move to a better place
            # can just hard code if it 
            if add_feat['type'] == 'org':
                corr12 = corr_func_org(input_1_modified, input_2_modified)
                corr21 = corr_func_org(input_2_modified, input_1_modified)
            elif add_feat['type'] == 'vfa':
                corr12 = corr_func_vfa(input_1_modified, input_2_modified)
                corr21 = corr_func_vfa(input_2_modified, input_1_modified)
            else:
                raise ValueError(f"self.add_feat: {self.add_feat} is not recognized")
                
            # corr12 = corr_func(input_1_modified, input_2_modified)
            # corr21 = corr_func(input_2_modified, input_1_modified)
            
            forward_combined_input = torch.cat((corr12, corr21), dim=1,)
            reverse_combined_input = torch.cat((corr21, corr12), dim=1,)

        forward_atomic = self._extract_atomic_transformation(forward_combined_input,)
        reverse_atomic = self._extract_atomic_transformation(reverse_combined_input,)
        return forward_atomic, reverse_atomic

    def forward(
        self,
        features_1: Tensor,
        features_2: Tensor,
        add_feat = None, # added by Hengjie
        # sum_diff: bool = True, # added by Hengjie
    ) -> tuple[GridComposableMapping, GridComposableMapping]:
        """Generate affine transformation parameters

        Args:
            features_1: Tensor with shape (batch_size, n_features, *volume_shape)
            features_2: Tensor with shape (batch_size, n_features, *volume_shape)

        Returns:
            Forward mapping
            Inverse mapping
            Optional regularization
        """
        forward_atomic, reverse_atomic = self._extract_atomic_transformations(
            features_1=features_1,
            features_2=features_2,
            add_feat=add_feat, # added by Hengjie
            # sum_diff=sum_diff, # added by Hengjie
        )
        device = features_1.device
        inverse_forward_atomic = self._invert_mapping(forward_atomic, device=device)
        inverse_reverse_atomic = self._invert_mapping(reverse_atomic, device=device)
        forward_transformation = forward_atomic @ inverse_reverse_atomic
        inverse_transformation = reverse_atomic @ inverse_forward_atomic
        return forward_transformation, inverse_transformation


class _DenseExtractionNetwork_my_v1(_BaseTransformationExtractionNetwork_my_v1):
    """
    List of methods:
        __init__
        _extract_atomic_transformation
        _invert_mapping
        _get_control_point_upper_bound
        _build_residual_block
            depend on _ResidualBlock class
    """
    def __init__(
        self,
        # n_input_features: int,
        # n_features: int,
        # n_convolutions: int,
        feature_coordinate_system: CoordinateSystem,
        transformation_coordinate_system: CoordinateSystem,
        forward_fixed_point_solver: FixedPointSolver,
        backward_fixed_point_solver: FixedPointSolver,
        max_control_point_multiplier: float,
        # activation_factory: IActivationFactory,
        # normalizer_factory: INormalizerFactory,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        n_convolutions: int = 2,
        res_skip: bool = True,
        act_name: tuple = ("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
        norm_name: tuple = ("group", {"num_groups": 4, "affine": True}),
        use_flowconv: bool = True,
        flowconv_type: str = None,
        add_feat=None,
    ) -> None:
        
        super().__init__()

        ### network building blocks
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
        
        
        # self._n_dims = len(transformation_coordinate_system.spatial_shape)
        # 
        # self.convolutions = ConvBlockNd(
        #     n_convolutions=n_convolutions,
        #     n_input_channels=2 * n_input_features,
        #     n_output_channels=n_features,
        #     kernel_size=(3,) * self._n_dims,
        #     padding=1,
        #     activation_factory=activation_factory,
        #     normalizer_factory=normalizer_factory,
        # )
        # self.final_convolution = ConvNd(
        #     n_input_channels=n_features,
        #     n_output_channels=self._n_dims,
        #     kernel_size=(1,) * self._n_dims,
        #     padding=0,
        #     bias=True,
        # )


        ### fixed point solver for inverting the DVF
        self._forward_fixed_point_solver = forward_fixed_point_solver
        self._backward_fixed_point_solver = backward_fixed_point_solver
        ### coordinate systmes
        self._feature_coordinate_system = feature_coordinate_system
        self._transformation_coordinate_system = transformation_coordinate_system
        
        ### max_control_point_value
        upsampling_factor_float = (
            feature_coordinate_system.grid_spacing_cpu()
            / transformation_coordinate_system.grid_spacing_cpu()
        )
        upsampling_factor = upsampling_factor_float.round().to(dtype=long).tolist()
        
        self._max_control_point_value = (
            max_control_point_multiplier
            * self._get_control_point_upper_bound(
                upsampling_factor,
            )
        )

    def _extract_atomic_transformation(
        self,
        combined_input: Tensor,
    ) -> GridComposableMapping:
        # output = self.convolutions(combined_input)
        # output = self.final_convolution(output)
        output = self.res_block(combined_input)
        output = self.out_block(output)
        output = self._max_control_point_value * tanh(output)
        return samplable_volume(
            output,
            coordinate_system=self._feature_coordinate_system,
            data_format=DataFormat.voxel_displacements(),
            sampler=CubicSplineSampler(
                prefilter=False, mask_extrapolated_regions=False
            ),
        ).resample_to(self._transformation_coordinate_system)

    def _invert_mapping(
        self, mapping: GridComposableMapping, device: torch_device
    ) -> GridComposableMapping:
        return mapping.invert(
            fixed_point_inversion_arguments={
                "forward_solver": self._forward_fixed_point_solver,
                "backward_solver": self._backward_fixed_point_solver,
            }
        ).resample()

    def _get_control_point_upper_bound(self, upsampling_factor: Sequence[int]) -> float:
        if tuple(upsampling_factor) in self.CONTROL_POINT_UPPER_BOUNDS_LOOKUP:
            return self.CONTROL_POINT_UPPER_BOUNDS_LOOKUP[tuple(upsampling_factor)]
        logger.info(
            "Computing cubic b-spline control point upper bound for upsampling factor %s "
            "which is not found in the lookup table. This might take a while. If you need "
            "this often, consider adding the upsampling factor to the lookup table.",
            tuple(upsampling_factor),
        )
        return compute_max_control_point_value(
            upsampling_factors=upsampling_factor,
            dtype=float64,
        ).item()

    # Precalculated upper bounds for the most common cases
    CONTROL_POINT_UPPER_BOUNDS_LOOKUP = {
        (1, 1, 1): 0.36,
        (2, 2, 2): 0.3854469526363808,
        (4, 4, 4): 0.39803114051999844,
        (8, 8, 8): 0.4007250760586097,
        (16, 16, 16): 0.40175329749498856,
        (32, 32, 32): 0.40253633025170044,
        (64, 64, 64): 0.4029187201371043,
        (128, 128, 128): 0.4031077300385704,
        (256, 256, 256): 0.4032092148874895,
        (512, 512, 512): 0.4032603042953307,
        (1, 1): 0.44999999999999996,
        (2, 2): 0.4923274169638207,
        (4, 4): 0.47980320571640545,
        (8, 8): 0.4845970764531083,
        (16, 16): 0.48612715818186963,
        (32, 32): 0.48728730908780815,
        (64, 64): 0.4879642422739664,
        (128, 128): 0.48831189411920906,
        (256, 256): 0.48848723983607806,
        (512, 512): 0.48857585860314345,
        (1024, 1024): 0.48862023712123526,
        (2048, 2048): 0.48864249909355595,
        (4096, 4096): 0.4886536264200244,
        (8192, 8192): 0.48865919697612337,
    }

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