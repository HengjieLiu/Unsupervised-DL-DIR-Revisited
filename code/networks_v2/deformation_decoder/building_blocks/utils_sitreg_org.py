

"""
List of classes:
    from https://github.com/honkamj/SITReg/blob/main/src/model/sitreg/model.py
        MappingPair
        _MappingBuilder
        _BaseTransformationExtractionNetwork
        _DenseExtractionNetwork
        
"""

# Standard library imports
from typing import NamedTuple, Sequence
from abc import abstractmethod
import logging

# PyTorch and related imports
import torch
from torch import Tensor, cat, tanh, long, float64
from torch.nn import Module

# Custom or project-specific imports – adjust the module names as needed
from your_module import (
    GridComposableMapping,    # The grid mapping type used for deformation operations.
    Identity,                 # A class that creates identity mappings with coordinate assignment.
    ConvBlockNd,              # Convolutional block used in the dense network.
    ConvNd,                   # Convolution layer used for the final convolution.
    samplable_volume,         # Function that wraps a tensor as a volume that can be sampled.
    DataFormat,               # Provides data format specifications (e.g., voxel displacements).
    CubicSplineSampler,       # A sampler used to perform cubic spline interpolation.
    compute_max_control_point_value,  # Computes the maximum control point value.
    CoordinateSystem,         # Represents a coordinate system for volumes.
    FixedPointSolver,         # Solver for fixed-point inversion of mappings.
    IActivationFactory,       # Factory for creating activation functions.
    INormalizerFactory,       # Factory for creating normalization layers.
)


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


class _BaseTransformationExtractionNetwork(Module):
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
    ) -> tuple[GridComposableMapping, GridComposableMapping]:
        input_1_modified = self._modify_input(features_1)
        input_2_modified = self._modify_input(features_2)
        forward_combined_input = cat(
            (input_1_modified - input_2_modified, input_1_modified + input_2_modified),
            dim=1,
        )
        reverse_combined_input = cat(
            (input_2_modified - input_1_modified, input_1_modified + input_2_modified),
            dim=1,
        )
        forward_atomic = self._extract_atomic_transformation(
            forward_combined_input,
        )
        reverse_atomic = self._extract_atomic_transformation(
            reverse_combined_input,
        )
        return forward_atomic, reverse_atomic

    def forward(
        self,
        features_1: Tensor,
        features_2: Tensor,
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
        )
        device = features_1.device
        inverse_forward_atomic = self._invert_mapping(forward_atomic, device=device)
        inverse_reverse_atomic = self._invert_mapping(reverse_atomic, device=device)
        forward_transformation = forward_atomic @ inverse_reverse_atomic
        inverse_transformation = reverse_atomic @ inverse_forward_atomic
        return forward_transformation, inverse_transformation


class _DenseExtractionNetwork(_BaseTransformationExtractionNetwork):
    def __init__(
        self,
        n_input_features: int,
        n_features: int,
        n_convolutions: int,
        feature_coordinate_system: CoordinateSystem,
        transformation_coordinate_system: CoordinateSystem,
        forward_fixed_point_solver: FixedPointSolver,
        backward_fixed_point_solver: FixedPointSolver,
        max_control_point_multiplier: float,
        activation_factory: IActivationFactory,
        normalizer_factory: INormalizerFactory,
    ) -> None:
        super().__init__()
        self._n_dims = len(transformation_coordinate_system.spatial_shape)
        upsampling_factor_float = (
            feature_coordinate_system.grid_spacing_cpu()
            / transformation_coordinate_system.grid_spacing_cpu()
        )
        upsampling_factor = upsampling_factor_float.round().to(dtype=long).tolist()
        self.convolutions = ConvBlockNd(
            n_convolutions=n_convolutions,
            n_input_channels=2 * n_input_features,
            n_output_channels=n_features,
            kernel_size=(3,) * self._n_dims,
            padding=1,
            activation_factory=activation_factory,
            normalizer_factory=normalizer_factory,
        )
        self.final_convolution = ConvNd(
            n_input_channels=n_features,
            n_output_channels=self._n_dims,
            kernel_size=(1,) * self._n_dims,
            padding=0,
            bias=True,
        )
        self._feature_coordinate_system = feature_coordinate_system
        self._transformation_coordinate_system = transformation_coordinate_system
        self._forward_fixed_point_solver = forward_fixed_point_solver
        self._backward_fixed_point_solver = backward_fixed_point_solver
        self._max_control_point_value = (
            max_control_point_multiplier
            * self._get_control_point_upper_bound(
                upsampling_factor,
            )
        )

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

    def _extract_atomic_transformation(
        self,
        combined_input: Tensor,
    ) -> GridComposableMapping:
        output = self.convolutions(combined_input)
        output = self.final_convolution(output)
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