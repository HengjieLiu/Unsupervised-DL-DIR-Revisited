# deformation_decoder/decoders/deformation_decoder_mysitreg.py




######### from SITReg
import torch
from torch import nn, Tensor
# from torch.nn import Module, nn.ModuleList
import torch.nn as nn
from typing import Sequence, cast
from logging import getLogger

### composable_mapping
from composable_mapping import (
    CoordinateSystem, #
    # CubicSplineSampler,
    # DataFormat,
    GridComposableMapping, #
    Identity, #
    LinearInterpolator, #
    OriginalFOV, #
    Start, #
    # affine,
    default_sampler, #
    samplable_volume, #
)

### deformation_inversion_layer
from deformation_inversion_layer.interface import FixedPointSolver


######### my customized imports
from ..registries import register_deformation_decoder

from ..building_blocks.utils_sitreg_v1 import MappingPair, _MappingBuilder
from ..building_blocks.utils_sitreg_v1 import _DenseExtractionNetwork_my_v1
# from ..building_blocks.block_utils import FlowConv


### utils_warp not needed in this script
# import sys
# sys.path.insert(0, '/homebase/DL_projects/wavereg/code')
# import utils_warp  # Assumed to provide SpatialTransformer, ComposeDVF, dvf_upsample


logger = getLogger(__name__)

# make this global for readability
dp_block_filtered_keys = [
    'kernel_size', 'bias', 
    'n_convolutions', 'res_skip', 
    'act_name', 'norm_name', 
    'use_flowconv', 'flowconv_type',
    # 'add_feat', ########$$$$  Note this is not needed here, I know this seems weird, but it is true!
]

########$$$$$$$$
# The key is that add_feat is not passed to the init method, it is directly passed in forward:
# return dense_extraction_network(
#     transformed_features_1,
#     transformed_features_2,
#     self.add_feat, # added by Hengjie
#     # self.sum_diff, # added by Hengjie
# )


@register_deformation_decoder("pyramidal_cnn_sitreg")
class DeformationDecoderMySITReg(nn.Module):

    """
    Note:
        the reverse happens in the forward method, not the __init__ method as contrary to other scripts (e.g., DeformationDecoderPyramidalCNN)
    
    =============================================== TO DO ===============================================
        I need to check if I can implement return_intermediates in the forward method !!!
    ======================================================================================================
    """
    #############################################################################################################
    ### Original comments
    #############################################################################################################
    """SITReg is a deep learning intra-modality image registration arhitecture
    fulfilling strict symmetry properties

    The implementation is dimensionality agnostic but PyTorch linear interpolation
    supports only 2 and 3 dimensional volumes.

    Arguments:
        feature_extractor: Multi-resolution feature extractor
        n_transformation_features_per_resolution: Defines how many features to
            use for extracting transformation in anti-symmetric update for each
            resolution. If None is given for some resolution, no deformation is
            extracted for that.
        n_transformation_convolutions_per_resolution: Defines how many convolutions to
            use for extracting transformation in anti-symmetric update for each
            resolution. If None is given for some resolution, no deformation is
            extracted for that.
        [REMOVED] affine_transformation_type: Defines which type of affine transformation
            to predict. If None, no affine transformation is predicted.
        input_voxel_size: Voxel size of the inputs images
        input_shape: Shape of the input images
        transformation_downsampling_factor: Downsampling factor for each
            dimension for the final deformation, e.g., providing [1.0, 1.0, 1.0] means
            no downsampling for three dimensional inputs.
        forward_fixed_point_solver: Defines fixed point solver for the forward pass
            of the deformation inversion layers.
        backward_fixed_point_solver: Defines fixed point solver for the backward pass
            of the deformation inversion layers.
        max_control_point_multiplier: Optimal maximum control point values are
            multiplied with this value to ensure that the individual
            deformations are invertible even after numerical errors. This should
            be some value just below 1, e.g. 0.99.
        activation_factory: Activation function to use.
        normalizer_factory: Normalizer factory to use. If None, no normalization
    """

    def __init__(
        self,
        decoder_params,
        forward_fixed_point_solver: FixedPointSolver,
        backward_fixed_point_solver: FixedPointSolver,
        max_control_point_multiplier: float,
        # feature_shapes, # previously use feature_extractor.get_shapes(), now i should calculate based on decoder_params
        # sum_diff: bool = True,
        # feature_extractor: FeatureExtractor,
        # n_transformation_features_per_resolution: Sequence[int],
        # n_transformation_convolutions_per_resolution: Sequence[int],
        # input_voxel_size: Sequence[float],
        # input_shape: Sequence[int],
        # transformation_downsampling_factor: Sequence[float],
        # forward_fixed_point_solver: FixedPointSolver,
        # backward_fixed_point_solver: FixedPointSolver,
        # max_control_point_multiplier: float,
        # activation_factory: IActivationFactory,
        # normalizer_factory: INormalizerFactory | None,
    ) -> None:
        
        super().__init__()

        ### my variables
        # add self?
        n_levels = decoder_params["n_levels"]
        
        input_shape = decoder_params["img_size"] # [160, 224, 192]
        input_voxel_size = [1.0, 1.0, 1.0] # decoder_params["input_voxel_size"] # [1.0, 1.0, 1.0]
        transformation_downsampling_factor= [1.0, 1.0, 1.0] # decoder_params["transformation_downsampling_factor"] # [1.0, 1.0, 1.0]

        # n_transformation_features_per_resolution = decoder_params["n_features_per_level"][::-1] # need to reverse
        # n_transformation_convolutions_per_resolution = [decoder_params["n_convolutions"]] * n_levels
        self.n_input_features_per_level = decoder_params["n_input_features_per_level"]
        self.n_features_per_level = decoder_params["n_features_per_level"]
        # self.n_convolutions = decoder_params["n_convolutions"]
        
        # feature_shapes = feature_extractor.get_shapes()
        # feature_shapes = feature_shapes # [[12, 160, 224, 192], [32, 80, 112, 96], [64, 40, 56, 48], [128, 20, 28, 24], [128, 10, 14, 12], [128, 5, 7, 6]]
        H, W, D = input_shape
        feature_shapes = [[c, H // (2 ** i), W // (2 ** i), D // (2 ** i)] for i, c in enumerate(self.n_input_features_per_level)]
        
        # downsampling_factors = feature_extractor.get_downsampling_factors()
        downsampling_factors = [[float(2**d)]*3 for d in range(n_levels)] # [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [8.0, 8.0, 8.0], [16.0, 16.0, 16.0], [32.0, 32.0, 32.0]]
        
        predict_dense_for_stages = [True] * n_levels
        self._predict_dense_for_stages = predict_dense_for_stages
        self._dense_prediction_indices = list(range(n_levels))
        # self._feature_extractor = feature_extractor


        ### !!! This is to be implemented for e.g. deformation_decoder_pyramidal_cnn as well !!!
        # if true the features would be summed and diffed before feeding into the network
        # if false it would stay the same
        # self.sum_diff = decoder_params["sum_diff"]
        self.add_feat = decoder_params["add_feat"]


        ### Get the coordinate systems
        # Build the list of coordinate systems one-by-one using a loop and indices
        coordinate_systems = []
        
        for i in range(len(predict_dense_for_stages)):
            if predict_dense_for_stages[i]:
                feature_shape = feature_shapes[i]
                downsampling_factor = downsampling_factors[i]
                cs = CoordinateSystem.centered_normalized(
                    spatial_shape=input_shape,
                    voxel_size=input_voxel_size,
                ).reformat(
                    downsampling_factor=downsampling_factor,
                    spatial_shape=feature_shape[1:],
                    reference=Start(),
                )
                coordinate_systems.append(cs)
        
        self._feature_coordinate_systems = nn.ModuleList(coordinate_systems)

        # self._feature_coordinate_systems = nn.ModuleList(
        #     [
        #         CoordinateSystem.centered_normalized(
        #             spatial_shape=input_shape,
        #             voxel_size=input_voxel_size,
        #         ).reformat(
        #             downsampling_factor=downsampling_factor,
        #             spatial_shape=feature_shape[1:],
        #             reference=Start(),
        #         )
        #         for (predict_deformation, feature_shape, downsampling_factor) in zip(
        #             predict_dense_for_stages,
        #             feature_shapes,
        #             feature_extractor.get_downsampling_factors(),
        #         )
        #         if predict_deformation
        #     ]
        # )
        self._transformation_coordinate_system = CoordinateSystem.centered_normalized(
            spatial_shape=input_shape,
            voxel_size=input_voxel_size,
        ).reformat(
            downsampling_factor=transformation_downsampling_factor,
            spatial_shape=OriginalFOV(fitting_method="ceil"),
        )
        self._image_coordinate_system = CoordinateSystem.centered_normalized(
            spatial_shape=input_shape,
            voxel_size=input_voxel_size,
        )


        self._affine_extraction_network = None

        ### Get the multi-level dense displacement prediction network ready
        # variable name for backward compatibility [I AM CONFUSED ABOUT WHAT THIS MEANS]
        
        # Filter keys for DP block parameters.
        dp_block_kwargs = {k: decoder_params[k] for k in dp_block_filtered_keys if k in decoder_params}
        
        not_none_dense_extraction_networks = []
        for i in range(len(self._dense_prediction_indices)):
            dense_prediction_index = self._dense_prediction_indices[i]
            if dense_prediction_index is not None:
                # feature_shape = feature_shapes[i]
                # n_transformation_features = n_transformation_features_per_resolution[i]
                # n_transformation_convolutions = n_transformation_convolutions_per_resolution[i]

                # feature_coordinate_system = cast(CoordinateSystem, self._feature_coordinate_systems[dense_prediction_index])
                feature_coordinate_system = cast(CoordinateSystem, self._feature_coordinate_systems[i])

                # in_channels = self.n_input_features_per_level[i] * 2
                if self.add_feat is None or self.add_feat['name'] == 'diffsum':
                    in_channels = self.n_input_features_per_level[i] * 2
                # elif self.add_feat['name'] == 'diff':
                #     in_channels = self.n_input_features_per_level[i] * 3
                # elif self.add_feat['name'] == 'diffonly':
                #     in_channels = self.n_input_features_per_level[i] * 1
                elif self.add_feat['name'] == 'corr':
                    in_channels = self.n_input_features_per_level[i] * 2 + 27 * 2 # note for bidir models, corr is also bidir 27*2
                elif self.add_feat['name'] == 'corronly':
                    in_channels = 27 * 2 # note for bidir models, corr is also bidir 27*2
                else:
                    raise ValueError(f"self.add_feat: {self.add_feat} is not recognized")
                    
                out_channels = self.n_features_per_level[i]
                
                net = _DenseExtractionNetwork_my_v1(
                    feature_coordinate_system=feature_coordinate_system,
                    transformation_coordinate_system=self._transformation_coordinate_system,
                    forward_fixed_point_solver=forward_fixed_point_solver,
                    backward_fixed_point_solver=backward_fixed_point_solver,
                    max_control_point_multiplier=max_control_point_multiplier,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **dp_block_kwargs,
                )
                
                # net = _DenseExtractionNetwork(
                #     n_input_features=feature_shape[0],
                #     n_features=n_transformation_features,
                #     n_convolutions=n_transformation_convolutions,
                #     feature_coordinate_system=cast(
                #         CoordinateSystem,
                #         self._feature_coordinate_systems[dense_prediction_index],
                #     ),
                #     transformation_coordinate_system=self._transformation_coordinate_system,
                #     forward_fixed_point_solver=forward_fixed_point_solver,
                #     backward_fixed_point_solver=backward_fixed_point_solver,
                #     max_control_point_multiplier=max_control_point_multiplier,
                #     activation_factory=activation_factory,
                #     normalizer_factory=normalizer_factory,
                # )
                
                not_none_dense_extraction_networks.append(net)
        
        self._not_none_dense_extration_networks = nn.ModuleList(not_none_dense_extraction_networks)
    
    @property
    def image_coordinate_system(self) -> CoordinateSystem:
        """Coordinate system used by the network for the inputs."""
        return self._image_coordinate_system

    @property
    def transformation_coordinate_system(self) -> CoordinateSystem:
        """Coordinate system used by the predicted transformations."""
        return self._transformation_coordinate_system

    def _extract_dense(
        self,
        batch_combined_features: torch.Tensor,
        coordinate_system: CoordinateSystem,
        mapping_builder: "_MappingBuilder",
        dense_extraction_network: nn.Module,
    ) -> tuple[GridComposableMapping, GridComposableMapping]:
        features_1, features_2 = torch.chunk(batch_combined_features, 2)
        transformed_features_1 = (
            (
                samplable_volume(
                    data=features_1,
                    coordinate_system=coordinate_system,
                )
                @ mapping_builder.left_forward()
            )
            .sample()
            .generate_values()
        )
        transformed_features_2 = (
            (
                samplable_volume(
                    data=features_2,
                    coordinate_system=coordinate_system,
                )
                @ mapping_builder.right_forward()
            )
            .sample()
            .generate_values()
        )
        return dense_extraction_network(
            transformed_features_1,
            transformed_features_2,
            self.add_feat, # added by Hengjie
            # self.sum_diff, # added by Hengjie
        )

    def _to_level_index(self, low_first_level_index: int) -> int:
        """Obtain high resolution first level index

        Args:
            low_first_level_index: Low resolution first index, -1 refers to affine level
        """
        n_dense_levels = sum(self._predict_dense_for_stages)
        return n_dense_levels - 1 - low_first_level_index

    def forward(
        self,
        batch_combined_features_list: list[torch.Tensor],
        return_intermediates: bool = False,
        resample_when_composing: bool = True,
        # mappings_for_levels: Sequence[tuple[int, bool]] = ((0, True),),
        # mappings_for_levels: Sequence[tuple[int, bool]] = ((0, False),),        
        # image_1: torch.Tensor,
        # image_2: torch.Tensor,
    ) -> list[MappingPair]:
        """Generate deformations

        Args:
            image_1: Image registered to image_2, torch.Tensor with shape
                (batch_size, n_channels, dim_1, ..., dim_{n_dims})
            image_2: Image registered to image_1, torch.Tensor with shape
                (batch_size, n_channels, dim_1, ..., dim_{n_dims})
            deformations_for_levels: Defines for which resolution levels the
                deformations are returned. The first element of each tuple is
                the index of the level and the second element indicates whether
                to include affine transformation in the deformation. Indexing
                starts from the highest resolution. Default value is [(0, True)]
                which corresponds to returning the full deformation at the
                highest resolution.

        Returns:
            List of MappingPairs in the order given by the input argument
            "mappings_for_levels"
        """
        with default_sampler(LinearInterpolator(mask_extrapolated_regions=False)):

            ### HJ DEBUGGING
            # print('len(batch_combined_features_list): ', len(batch_combined_features_list))
            # for i, batch_combined_features in enumerate(batch_combined_features_list):
            #     print(i, batch_combined_features.shape)

            n_levels = len(batch_combined_features_list)

            # build mappings_for_levels from return_intermediates
            # print(f"return_intermediates: {return_intermediates}") ### HJ DEBUGGING
            if return_intermediates:
                # mappings_for_levels = ((d, False) for d in range(n_levels)) # won't work
                mappings_for_levels = [(d, False) for d in range(n_levels)]
            else:
                mappings_for_levels = ((0, False),)
            mappings_for_levels_set = set(mappings_for_levels)
            
            # batch_combined_features_list: list[torch.Tensor] = self._feature_extractor((image_1, image_2))
            
            # Removed affine extraction.
            # Instead, we always start with an identity mapping.
            # dtype = image_1.dtype
            # device = image_1.device
            dtype = batch_combined_features_list[0].dtype
            device = batch_combined_features_list[0].device
            
            identity = Identity(dtype=dtype, device=device).assign_coordinates(
                self._transformation_coordinate_system
            )
            mapping_builder = _MappingBuilder(
                forward_affine=identity,
                inverse_affine=identity,
                resample_when_composing=resample_when_composing,
            )


            ### get all output_mappings
            output_mappings: dict[tuple[int, bool], MappingPair] = {}
            for low_first_level_index, (
                batch_combined_features,
                dense_prediction_index,
            ) in enumerate(
                zip(
                    reversed(batch_combined_features_list),
                    reversed(self._dense_prediction_indices),
                )
            ):
                if dense_prediction_index is not None:
                    logger.debug(
                        "Starting deformation extraction from features with shape %s",
                        tuple(batch_combined_features.shape),
                    )
                    
                    coordinate_system=cast(CoordinateSystem, self._feature_coordinate_systems[dense_prediction_index])
                    dense_extraction_network=self._not_none_dense_extration_networks[dense_prediction_index]
                    
                    forward_dense, inverse_dense = self._extract_dense(
                        batch_combined_features=batch_combined_features,
                        coordinate_system=coordinate_system,
                        mapping_builder=mapping_builder,
                        dense_extraction_network=dense_extraction_network,
                    )
                    mapping_builder.update(forward_dense=forward_dense, inverse_dense=inverse_dense)
                
                level_index = self._to_level_index(low_first_level_index)

                include_affine = False
                output_mappings[(level_index, include_affine)] = (mapping_builder.as_mapping_pair(include_affine))

                del batch_combined_features_list[-1]

            ### HJ DEBUGGING
            # for i, batch_combined_features in enumerate(batch_combined_features_list):
            #     print(i, batch_combined_features)
            # for i, mapping_index in enumerate(mappings_for_levels):
            #     print(i, mapping_index)
            # print('output_mappings.keys(): ', output_mappings.keys())
            # print('mappings_for_levels: ', mappings_for_levels)

            list_mappings = [output_mappings[mapping_index] for mapping_index in mappings_for_levels]
            list_disps = [(map_.forward_mapping._data.displacements, map_.inverse_mapping._data.displacements) for map_ in list_mappings]

            disp_forward = output_mappings[(0, False)].forward_mapping._data.displacements
            disp_inverse = output_mappings[(0, False)].inverse_mapping._data.displacements
            
            return disp_forward, disp_inverse, list_disps, list_mappings

            # original return
            # return [output_mappings[mapping_index] for mapping_index in mappings_for_levels]




        