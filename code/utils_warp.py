"""
General description:
    Functions used for image warping using DVF (deformation vector field), and DVF handling

Function/Class list:
    DVF warping and DVF composing:
        SpatialTransformer
        ComposeDVF
    Upsampling and downsampling functions for image and DVF:
        dvf_upsample
        vol_downsamplex2
        ResizeFlow
            [Future work: can use ResizeFlow to replace dvf_upsample]
            already done in:
                deformation_decoder/decoders/
                    deformation_decoder_pyramidal_cnn.py
                    deformation_decoder_pyramidal_vfa.py
"""

import warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf


####################################################################################################
### PART 1: DVF warping and DVF composing 
####################################################################################################

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer (N = 2 or 3)
        warps an image using a deformation vector field (DVF), also called dense displacement field (DDF)
        (note in DDF convention, DVF is short for dense velocity field)
    Based on:
        VoxelMorph: https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/layers.py: class SpatialTransformer(nn.Module)
        LapIRN: https://github.com/cwmok/LapIRN/blob/master/Code/miccai2020_model_stage.py: class SpatialTransform_unit(nn.Module)
        monai: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/blocks/warp.py: class Warp(nn.Module)
    Modifications:
        1. Add indexing='ij' to torch.meshgrid
            see: https://pytorch.org/docs/stable/generated/torch.meshgrid.html
        2. Add all arguments of torch.nn.functional.grid_sample() to the forward method:
            grid_sample(input, grid, mode, padding_mode, align_corners)
            see ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                mode: use 'bilinear' for image warping, and 'nearest' for segmentation warping
                padding_mode: default to 'border' to avoid black borders
                align_corners: default to True
        3. Register buffer: 
             add persistent=persistent, default to False
             this can avoid the grid to pollute the state_dict, resulting in much smaller model files (.pt)
             see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
                 https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict/61668/3
                 Actually someone has proposed this as a pull for vxm already: https://github.com/voxelmorph/voxelmorph/pull/349
    Notes:
        The reason to reverse channel (new_locs = new_locs[..., [2, 1, 0]]) is described in:
            https://github.com/voxelmorph/voxelmorph/issues/213
            https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997/9
    """

    def __init__(self, size: Tuple[int, ...], persistent: bool = False) -> None:
        super().__init__()

        if len(size) not in [2, 3]:  # Handle both 2D and 3D cases
            raise ValueError("Only 2D and 3D inputs are supported.")
        
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)

        grid = torch.unsqueeze(grid, 0) # Add batch dimension
        grid = grid.type(torch.FloatTensor)

        # Register the grid buffer
        # Registering the grid as a buffer cleanly moves it to the GPU, but it also pullutes the state_dict.
        # Use persistent=False to avoid polluting the state_dict.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid, persistent=persistent)

    def forward(self, src: torch.FloatTensor, flow: torch.FloatTensor,
                mode: str = 'bilinear', padding_mode: str = 'border', align_corners: bool = True) -> torch.FloatTensor:
        
        if src.ndim == 5:  # 5D tensor for 3D image
            if src.shape[-3:] != flow.shape[-3:]:
                raise ValueError(f"Mismatch in last 3 dimensions between src and flow: {src.shape[-3:]} != {flow.shape[-3:]}")
        elif src.ndim == 4:  # 4D tensor for 2D image
            if src.shape[-2:] != flow.shape[-2:]:
                raise ValueError(f"Mismatch in last 2 dimensions between src and flow: {src.shape[-2:]} != {flow.shape[-2:]}")
        else:
            raise ValueError("Input tensor must be either 4D (2D images) or 5D (3D images)!")


        # new locations: compute new locations by adding flow to the grid
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed (resolved, see Notes)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        
        # perform warping using grid_sample
        warped = nnf.grid_sample(src, new_locs, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        
        return warped


class ComposeDVF(nn.Module):
    """
    N-D DVF Composer (N = 2 or 3)
        Compose two displacement fields, return the displacement that warps by u1 followed by u2.
    Notes:
        Composing multiple DVFs before warping reduces the blurring effect caused by multiple interpolations.
        warp(img, compose(u1, u2)) ~= warp(warp(img, u1), u2),
            where warp is the SpatialTransformer.   
    Ref:
        https://github.com/ebrahimebrahim/abcd-registration-experiments/blob/58b1b7e5118729ad22fe6ba8bbcf87c252af26eb/util.py#L255-L263: class ComposeDDF(nn.Module)
        https://github.com/JinxLv/reimplemention-of-Dual-PRNet/blob/83f2e42ac6316fac49708a7bc42b303a8ff16af2/network/base_networks.py#L226
    """

    def __init__(self, size: Tuple[int, ...]) -> None:
        super().__init__()

        self.transformer = SpatialTransformer(size)

    def forward(self, u1: torch.FloatTensor, u2: torch.FloatTensor,
                mode: str = 'bilinear', padding_mode: str = 'border', align_corners: bool = True) -> torch.FloatTensor:
        return u2 + self.transformer(u1, u2, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


class ComposeDVF_list(nn.Module):
    """
    Compose an arbitrary number of displacement fields (DVFs) provided in a list.

    When you call:
         composed_dvf = compose([u1, u2, u3])
    it returns a displacement field that, when used for warping an image, is
    approximately equivalent to:
         warp(warp(warp(img, u1), u2), u3)
         
    Note:
         - The DVFs must be provided in the order they are applied (i.e. u1 is applied first).
         - The composition is computed in reverse order internally to preserve the correct application order.

    unittest:
        /homebase/DL_projects/wavereg/code/unittestwarp/composer_250210.ipynb
    """
    def __init__(self, size: Tuple[int, ...]) -> None:
        super().__init__()
        self.transformer = SpatialTransformer(size)

    def forward(self, dvfs: List[torch.FloatTensor],
                mode: str = 'bilinear', padding_mode: str = 'border', 
                align_corners: bool = True) -> torch.FloatTensor:
        if not isinstance(dvfs, list):
            raise TypeError("dvfs must be provided as a list of displacement fields.")
        if not dvfs:
            raise ValueError("The list of DVFs is empty. Provide at least one DVF.")

        # Start with the first DVF
        composed = dvfs[0]
        # Then iterate over the remaining DVFs in order
        for dvf in dvfs[1:]:
            composed = dvf + self.transformer(composed, dvf,
                                              mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        return composed
        
    # def forward(self, dvfs: List[torch.FloatTensor],
    #             mode: str = 'bilinear', padding_mode: str = 'border', 
    #             align_corners: bool = True) -> torch.FloatTensor:
    #     if not isinstance(dvfs, list):
    #         raise TypeError("dvfs must be provided as a list of displacement fields.")
    #     if not dvfs:
    #         raise ValueError("The list of DVFs is empty. Provide at least one DVF.")

    #     # If only one DVF is provided, return it directly.
    #     composed = dvfs[-1]

    #     # Iterate over the list in reverse order (excluding the last one),
    #     # composing the fields so that the first DVF is applied first.
    #     for dvf in reversed(dvfs[:-1]):
    #         composed = composed + self.transformer(dvf, composed,
    #                                                 mode=mode, 
    #                                                 padding_mode=padding_mode, 
    #                                                 align_corners=align_corners)
    #     return composed



class VecInt(nn.Module):
    """
    Version control:
        Currently minimally adapted from the voxelmorph code:
            https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/layers.py
    
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec




####################################################################################################
### PART 2: Upsampling and downsampling functions for image and DVF
####################################################################################################

def dvf_upsample(dvf: torch.FloatTensor, scale_factor=2.0, mode='trilinear') -> torch.FloatTensor:
    """
    Upsample the DVF by a scale factor
    Note: To maintain the unit of the DVF, the upsampled DVF should be multiplied by the scale factor
    """
    if dvf.ndim == 4:  # 4D tensor for 2D image
        mode = 'bilinear'
        
    upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
    dvf_upsampled = upsample(dvf) * scale_factor
    return dvf_upsampled


def vol_downsamplex2(vol: torch.FloatTensor) -> torch.FloatTensor:
    """
    Downsample the volume by a factor of 2 using average pooling
    Ref:
        from LapIRN/Code/miccai2020_model_stage.py:
        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    """
    if vol.ndim == 4:  # 4D tensor for 2D image
        down_scale_func = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    elif vol.ndim == 5:  # 5D tensor for 3D image
        down_scale_func = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    else:
        raise ValueError("Input must be either 4D (for 2D images) or 5D (for 3D volumes).")

    return down_scale_func(vol)


class ResizeFlow(nn.Module):
    # Based on: https://github.com/BailiangJ/rethink-reg/blob/6fc0af1f04a707bddbcfb5246e09e295d0b3a8fe/models/utils/resize_flow.py#L8
    """
    My modifications:
        F->nnf
        add warnings
    """
    
    
    """Resize and rescale a flow field.

    Args:
        spatial_scale (float): scaling factor of spatial resizing.
        flow_scale (float): scaling factor of flow components.
        ndim (int): number of dimensions.
    """
    def __init__(self, spatial_scale: float, flow_scale: float, ndim: int):
        super().__init__()
        
        # Warn if the scales differ by more than 1e-6
        if abs(spatial_scale - flow_scale) > 1e-6:
            warnings.warn(
                f"spatial_scale and flow_scale differ by more than 1e-6. "
                f"Values: spatial_scale={spatial_scale}, flow_scale={flow_scale}",
                UserWarning
            )
            
        self.spatial_scale = spatial_scale
        self.flow_scale = flow_scale
        if ndim == 2:
            self.interp_mode = 'bilinear'
        elif ndim == 3:
            self.interp_mode = 'trilinear'
        else:
            raise KeyError(f'Unsupported ndim for ResizeFlow:{ndim}.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flow_scale < 1:
            # resize first to save memory
            x = nnf.interpolate(x,
                              align_corners=True,
                              scale_factor=self.spatial_scale,
                              recompute_scale_factor=True,
                              mode=self.interp_mode)
            x = self.flow_scale * x

        elif self.flow_scale > 1:
            # multiply first to save memory
            x = self.flow_scale * x
            x = nnf.interpolate(x,
                              align_corners=True,
                              scale_factor=self.spatial_scale,
                              recompute_scale_factor=True,
                              mode=self.interp_mode)
        else:  # self.flow_scale = 1
            x = nnf.interpolate(x,
                              align_corners=True,
                              scale_factor=self.spatial_scale,
                              recompute_scale_factor=True,
                              mode=self.interp_mode)

        return x

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(spatial_scale={self.spatial_scale}, '
                     f'flow_scale={self.flow_scale})')
        return repr_str
