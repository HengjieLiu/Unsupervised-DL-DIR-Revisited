"""
List of functions and variables
    ### Helper functions
    numpy_overview
    torch_overview
    numpy2torch
    torch2numpy

    ### Data loading
    load_nifti_to_array
    reorient_OASIS_to_RAS
    process_dvf_greedy_oasis

    ### Deformation related
    SpatialTransformer
    mk_grid_img
    mk_grid_img_3d
    JDet

    ### Display functions
    get_slice_obj
    plot_registration_results

    ### Variables/Constants/Helper for display
    itksnap_lut_oasis
    n_labels_oasis
    colors_oasis_list
    cmap_seg_oasis
    boundaries_oasis
    norm_seg_oasis
    DEFAULT_IMSHOW_PARAMS
    DEFAULT_GRID_PARAMS
    DEFAULT_OVERLAY_PARAMS
    DEFAULT_CONTOUR_OVERLAY_PARAMS
    _get_default_disp_setting
    _get_num_data_subplots
"""

# Standard library imports
from typing import Tuple, Optional
# Scientific computing imports
import numpy as np
import scipy.ndimage
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as nnf
# Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable # For more flexible colorbars


####################################################################################################
### PART 1: Helper functions
####################################################################################################

def numpy_overview(array: np.ndarray, varname: str = None) -> None:
    """
    Print overview of a numpy array
    format: {varname}: {dtype}, {ndim}D, shape={shape}, min={min}, max={max}
    """
    if varname is not None:
        print(f'{varname}: {array.dtype}, {array.ndim}D, shape={array.shape}, min={np.min(array)}, max={np.max(array)}')
    else:
        print(f'{array.dtype}, {array.ndim}D, shape={array.shape}, min={np.min(array)}, max={np.max(array)}')

        
def torch_overview(tensor: torch.Tensor, varname: str = None) -> None:
    """
    Print overview of a torch tensor
    format: {varname}: {dtype}, {ndim}D, size={size}, device={device}, min={min}, max={max}
    """
    if varname is not None:
        print(f'{varname}: {tensor.dtype}, {tensor.ndim}D, size={tensor.size()}, device={tensor.device}, min={torch.min(tensor).item()}, max={torch.max(tensor).item()}')
    else:
        print(f'{tensor.dtype}, {tensor.ndim}D, size={tensor.size()}, device={tensor.device}, min={torch.min(tensor).item()}, max={torch.max(tensor).item()}')

        
def numpy2torch(array: np.ndarray, device=None, CHECK=True) -> torch.FloatTensor:
    """
    Convert numpy array to torch tensor
        If CHECK is False:
            return torch.from_numpy(array).float()
        If CHECK is True:
            Handle specific cases for 3D volumes and 4D DVFs.
                Inputs:  3D volume array [H,W,D] or 4D DVF array [C==3,H,W,D]
                Outputs: 5D tensor [N,C,H,W,D]
    """

    if CHECK:
        assert array.ndim in [3, 4], f'Input array should be 3D volume or 4D DVF, got {array.ndim}D.'
        if array.ndim == 3: # 3D volume (H,W,D)
            array = array[np.newaxis, np.newaxis, ...]  # Add batch and channel dims
        elif array.ndim == 4: # 4D DVF (3,H,W,D)
            assert array.shape[0] == 3, f'Expected 3 channels for 4D DVF, got {array.shape[0]}.'
            array = array[np.newaxis, ...]  # Add batch dim
    
    # convert to tensor
    tensor = torch.from_numpy(array).float()
    
    # move tensor to device if specified
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def torch2numpy(tensor: torch.FloatTensor, CHECK=True) -> np.ndarray:
    """
    Convert torch tensor to numpy array
        If CHECK is False:
            return tensor.detach().cpu().numpy()
        If CHECK is True:
            Handle specific cases for 3D volumes and 4D DVFs.
                Inputs:  5D tensor [N,C,H,W,D]
                    If C==1, it is a volume tensor, if C==3, it is a DVF tensor
                Outputs: 3D volume array [H,W,D] if C==1, or 4D DVF array [3,H,W,D] if C==3
    """

    assert tensor.ndim == 5, f'Expected 5D tensor, got {tensor.ndim}D.'

    if CHECK:
        assert tensor.size(1) in [1, 3], f'Expected channel size 1 for volume or 3 for DVF, got {tensor.size(1)}.'
        
    # Detach tensor and move to CPU if necessary
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert to numpy array
    array = tensor.numpy()

    # Handle squeezing based on channel dimension
    if CHECK:
        if tensor.size(1) == 1:
            array = np.squeeze(array, axis=(0, 1))  # Squeeze batch and channel for 3D volume
        else:
            array = np.squeeze(array, axis=0)  # Squeeze only batch for 4D DVF
        
    return array


####################################################################################################
### PART 2: Data loading
####################################################################################################

def load_nifti_to_array(filename: str, ret_affine: bool = False):
    """
    Load a NIfTI (.nii or .nii.gz) file and return its image data as a numpy array.

    Parameters:
    -----------
    file_path : str
        The path to the NIfTI file.

    Returns:
    --------
    np.ndarray
        The image data contained in the NIfTI file.
    """
    # Load the NIfTI file
    nii_img = nib.load(filename)
    
    # Extract the image data as a numpy array
    img_array = nii_img.get_fdata()
    
    if ret_affine:
        return img_array, nii_img.affine
    else:
        return img_array


def reorient_OASIS_to_RAS(arr):
    """
    Reorients an array from LIA+ to RAS+ (i.e., converts from OASIS to LUMIR convention).
    
    Process:
      - LIA+ -> LAI+   via transpose (swap 2nd and 3rd dimensions)
      - LAI+ -> RAI+   via flipping along axis 0
      - RAI+ -> RAS+   via flipping along axis 2
    """
    arr = np.transpose(arr, [0, 2, 1])
    arr = np.flip(arr, 0)
    arr = np.flip(arr, 2)
    return arr


def process_dvf_greedy_oasis(dvf_array, channel_last=True):
    """
    Process the DVF from greedy array by squeezing, transposing, fixing sign, and flipping.
    
    Parameters:
        dvf_array (np.ndarray): Input DVF array, for example with shape (160, 192, 224, 1, 3).
    
    Returns:
        np.ndarray: Processed DVF array with a copy of the data.
                    Expected shape after processing: (3, 160, 224, 192)
    """
    # Remove singleton dimensions (e.g., converting (160, 192, 224, 1, 3) to (160, 192, 224, 3))
    dvf_gd = dvf_array.squeeze()
    
    # Rearrange dimensions from (160, 192, 224, 3) to (160, 224, 192, 3)
    dvf_gd = np.transpose(dvf_gd, [0, 2, 1, 3])
    
    # Fix sign by multiplying with (-1, -1, 1)
    c_sign = (-1, -1, 1)
    dvf_gd = dvf_gd * np.array(c_sign)
    
    # Flip along the specified axes: (True, False, True)
    sp_flip = (True, False, True)
    for axis, flip_flag in enumerate(sp_flip):
        if flip_flag:
            dvf_gd = np.flip(dvf_gd, axis=axis)

    if not channel_last:
        # Rearrange dimensions to (3, 160, 224, 192)
        dvf_gd = np.transpose(dvf_gd, [3, 0, 1, 2])
    
    # Return a copy of the processed array
    return dvf_gd.copy()
    

####################################################################################################
### PART 3: Deformation related
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


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 224, 192)):
    """
    Direct copy from:
        https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/9b88e630398949bd6a429871fbc516e212334353/VoxelMorph/train_VoxelMorph.py#L171
    Note that the line_thickness might have some issue.
    """
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def mk_grid_img_3d(grid_step, line_thickness=1, grid_size=(160, 224, 192), axis_to_slice=0, to_tensor=True):
    """
    Creates a 3D grid image as a 5D numpy array or torch tensor (1, 1, H, W, D)
    For a give axis_to_slice in [0, 1, 2], 
        create grid lines perpendicular to the axis_to_slice,
        so when slicing along axis_to_slice, we will see the regular grid pattern.
        after warping the 3D grid image with a deformation field,
        we will see the deformed/warped grid pattern.

    Such displacement/deformation field visualization follows VoxelMorph's and TransMorph's convention.

    CAUTION that although the grid image is 3D, it is still a 2D grid pattern repeated along the axis_to_slice.
    And the deformation along the axis_to_slice is not visible in the grid image.

    Originally based on:
        https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/9b88e630398949bd6a429871fbc516e212334353/VoxelMorph/train_VoxelMorph.py#L171
        
    """
    if not isinstance(axis_to_slice, int) or axis_to_slice not in [0, 1, 2]:
        raise ValueError("axis_to_slice must be an integer: 0, 1, or 2.")

    grid_img_np = np.zeros(grid_size, dtype=np.float32)

    if axis_to_slice == 0:
        # Lines perpendicular to axis 1 (grid_img_np.shape[1])
        for val in range(0, grid_img_np.shape[1], grid_step):
            idx = val + line_thickness - 1
            if 0 <= idx < grid_img_np.shape[1]:
                grid_img_np[:, idx, :] = 1
        # Lines perpendicular to axis 2 (grid_img_np.shape[2])
        for val in range(0, grid_img_np.shape[2], grid_step):
            idx = val + line_thickness - 1
            if 0 <= idx < grid_img_np.shape[2]:
                grid_img_np[:, :, idx] = 1
    elif axis_to_slice == 1:
        # Lines perpendicular to axis 0 (grid_img_np.shape[0])
        for val in range(0, grid_img_np.shape[0], grid_step):
            idx = val + line_thickness - 1
            if 0 <= idx < grid_img_np.shape[0]:
                grid_img_np[idx, :, :] = 1
        # Lines perpendicular to axis 2 (grid_img_np.shape[2])
        for val in range(0, grid_img_np.shape[2], grid_step):
            idx = val + line_thickness - 1
            if 0 <= idx < grid_img_np.shape[2]:
                grid_img_np[:, :, idx] = 1
    elif axis_to_slice == 2:
        # Lines perpendicular to axis 0 (grid_img_np.shape[0])
        for val in range(0, grid_img_np.shape[0], grid_step):
            idx = val + line_thickness - 1
            if 0 <= idx < grid_img_np.shape[0]:
                grid_img_np[idx, :, :] = 1
        # Lines perpendicular to axis 1 (grid_img_np.shape[1])
        for val in range(0, grid_img_np.shape[1], grid_step):
            idx = val + line_thickness - 1
            if 0 <= idx < grid_img_np.shape[1]:
                grid_img_np[:, idx, :] = 1

    # Add batch and channel dimensions
    grid_img_5d_np = grid_img_np[None, None, ...]

    if to_tensor:
        grid_img = torch.from_numpy(grid_img_5d_np).cuda()
    else:
        grid_img = grid_img_5d_np

    return grid_img


class JDet:
    """

    
    Modified based on https://github.com/BailiangJ/rethink-reg/blob/main/models/metrics/sdlogjac.py
    """
    def __call__(self, disp: np.ndarray, fg_mask:Optional[np.ndarray] = None):
        '''
        Args:
            disp: displacement field of shape (B, 3, H, W, D)
            fg_mask: foreground mask of shape (1,1,H,W,D) or (B,1,H,W,D)
        '''
        B, _, H, W, D = disp.shape

        if fg_mask is None:
            fg_mask = np.ones((B, 1, H, W, D), dtype=np.float32)
        else:
            if fg_mask.shape[0] == 1:
                fg_mask = fg_mask.repeat(B, axis=0)
                # print(fg_mask.shape)
            fg_mask = fg_mask.astype(np.float32)
        fg_mask = fg_mask.squeeze(1)
        assert fg_mask.shape == (B, H, W, D)

        gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
        grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
        gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

        # Compute the gradient of the displacement field
        # gradx_disp: (B, 3, H, W, D)
        gradx_disp = np.stack([
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)
        ],
                              axis=1)

        # grady_disp: (B, 3, H, W, D)
        grady_disp = np.stack([
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)
        ],
                              axis=1)

        # gradz_disp: (B, 3, H, W, D)
        gradz_disp = np.stack([
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)
        ],
                              axis=1)

        # grad_disp: (B, 3, 3, H, W, D)
        grad_disp = np.stack([gradx_disp, grady_disp, gradz_disp], 1)

        # jacobian: (B, 3, 3, H, W, D)
        # displacement jacobian to deformation jacobian
        jacobian = grad_disp + np.eye(3, 3).reshape(1, 3, 3, 1, 1, 1)
        jacobian = jacobian[:, :, :, 2:-2, 2:-2, 2:-2]
        fg_mask = fg_mask[:, 2:-2, 2:-2, 2:-2]
        # jacdet: (B, H, W, D)
        jacdet = jacobian[:, 0, 0, ...] * \
                 (jacobian[:, 1, 1, ...] * jacobian[:, 2, 2, ...] -
                  jacobian[:, 1, 2, ...] * jacobian[:, 2, 1, ...]) - \
                 jacobian[:, 1, 0, ...] * \
                 (jacobian[:, 0, 1, ...] * jacobian[:, 2, 2, ...] -
                  jacobian[:, 0, 2, ...] * jacobian[:, 2, 1, ...]) + \
                 jacobian[:, 2, 0, ...] * \
                 (jacobian[:, 0, 1, ...] * jacobian[:, 1, 2, ...] -
                  jacobian[:, 0, 2, ...] * jacobian[:, 1, 1, ...])

        non_pos_jacdet = np.sum((jacdet <= 0)*fg_mask, axis=(1, 2, 3))

        log_jacdet = np.log((jacdet + 3).clip(0.000000001, 1000000000))

        # return np.std(log_jacdet, axis=(1, 2, 3)), non_pos_jacdet#, jacdet
        # return np.std(log_jacdet, axis=(1, 2, 3)), non_pos_jacdet, jacdet.squeeze()

        jdet = jacdet.squeeze()
        std_log_jacdet = np.std(log_jacdet, axis=(1, 2, 3))
        return jdet, non_pos_jacdet, std_log_jacdet


# --- Define the OASIS Look-Up Table (LUT) ---
# The colar values are trying to match the OASIS segmentation labels in ITK-SNAP.
itksnap_lut_oasis = {
    0:  (  0,   0,   0),   # Clear / Black
    1:  (255,   0,   0),   # Label 1 / Red
    2:  (  0, 255,   0),   # Label 2 / Lime
    3:  (  0,   0, 255),   # Label 3 / Blue
    4:  (255, 255,   0),   # Label 4 / Yellow
    5:  (  0, 255, 255),   # Label 5 / Cyan
    6:  (255,   0, 255),   # Label 6 / Magenta
    7:  (255, 239, 213),   # Label 7 / PapayaWhip
    8:  (  0,   0, 205),   # Label 8 / MediumBlue
    9:  (205, 133,  63),   # Label 9 / Peru
    10: (210, 180, 140),   # Label 10 / Tan
    11: (102, 205, 170),   # Label 11 / MediumAquamarine
    12: (  0,   0, 128),   # Label 12 / Navy
    13: (  0, 139, 139),   # Label 13 / DarkCyan
    14: ( 46, 139,  87),   # Label 14 / SeaGreen
    15: (255, 228, 225),   # Label 15 / MistyRose
    16: (106,  90, 205),   # Label 16 / SlateBlue
    17: (221, 160, 221),   # Label 17 / Plum
    18: (233, 150, 122),   # Label 18 / DarkSalmon
    19: (165,  42,  42),   # Label 19 / Brown
    20: (255, 250, 250),   # Label 20 / Snow
    21: (147, 112, 219),   # Label 21 / MediumPurple
    22: (218, 112, 214),   # Label 22 / Orchid
    23: ( 75,   0, 130),   # Label 23 / Indigo
    24: (255, 182, 193),   # Label 24 / LightPink
    25: ( 60, 179, 113),   # Label 25 / MediumSeaGreen
    26: (255, 235, 205),   # Label 26 / BlanchedAlmond
    27: (255, 228, 196),   # Label 27 / Bisque
    28: (218, 165,  32),   # Label 28 / GoldenRod
    29: (  0, 128, 128),   # Label 29 / Teal
    30: (188, 143, 143),   # Label 30 / RosyBrown
    31: (255, 105, 180),   # Label 31 / HotPink
    32: (255, 218, 185),   # Label 32 / PeachPuff
    33: (222, 184, 135),   # Label 33 / BurlyWood
    34: (127, 255,   0),   # Label 34 / Chartreuse
    35: (139,  69,  19)    # Label 35 / SaddleBrown
}

# --- Create the colormap and norm for OASIS ---
# Number of distinct labels (0 to 35 means 36 labels)
n_labels_oasis = 35
# Build the list of colors, normalized to 0-1
colors_oasis_list = []
for i in range(n_labels_oasis + 1): # Iterate from 0 to 35
    if i in itksnap_lut_oasis:
        colors_oasis_list.append(np.array(itksnap_lut_oasis[i]) / 255.0)
    else:
        # Handle missing labels, e.g., by adding a default color or raising an error
        print(f"Warning: Color for label {i} is not defined in itksnap_lut_oasis. Using white as default.")
        colors_oasis_list.append(np.array([1.0, 1.0, 1.0])) # Default to white

# Create the ListedColormap
cmap_seg_oasis = ListedColormap(colors_oasis_list, name='oasis_segmentation_labels')

# Create the BoundaryNorm
# Boundaries should be at k ± 0.5 to map integer k to colors_oasis_list[k]
# For n_labels_oasis = 35 (labels 0-35), we need 36 colors and 37 boundaries
# Boundaries will go from -0.5 to 35.5
boundaries_oasis = np.arange(-0.5, n_labels_oasis + 1.5, 1)
norm_seg_oasis = BoundaryNorm(boundaries=boundaries_oasis, ncolors=len(colors_oasis_list))


# --- Default Display Settings ---
DEFAULT_IMSHOW_PARAMS = {
    'cmap': 'gray', 
    'vmin': None, 'vmax': None,
    'origin': 'lower',
    'aspect': 'equal', 
    'interpolation': 'nearest', 'alpha': 1.0
}
DEFAULT_GRID_PARAMS = {'line_color': 'white', 'line_width': 0.5, 'density': 10, 'background_color': 'black'}
DEFAULT_OVERLAY_PARAMS = {'base_cmap': 'gray', 'overlay_cmap': 'viridis', 'overlay_alpha': 0.5}
DEFAULT_CONTOUR_OVERLAY_PARAMS = { # For overlaying contours on an image
    'show': False,
    'level': 0.0, # The value at which to draw the contour line
    'color': 'black', # Default color for the contour line
    'linewidth': 0.8, # Default linewidth
    'linestyle': '-'  # Default linestyle e.g., '-', '--', '-.', ':'
}


def _get_default_disp_setting():
    """Returns a deep copy of default display settings for a subplot."""
    return {
        'plot_type': 'imshow',
        'imshow_params': DEFAULT_IMSHOW_PARAMS.copy(),
        'grid_params': DEFAULT_GRID_PARAMS.copy(),
        'overlay_params': DEFAULT_OVERLAY_PARAMS.copy(),
        'contour_overlay': DEFAULT_CONTOUR_OVERLAY_PARAMS.copy(), # ADDED contour settings
        'show_axis': False,
        'custom_plot_func': None,
        'colorbar': {'show': False}  # Settings for an individual subplot colorbar
    }


def _get_num_data_subplots(category, num_methods):
    """Determines how many data subplots a row category should have."""
    if category == 'deformed':
        return num_methods + 2  # Fixed, Moving, Method1, ..., MethodM
    elif category == 'diff':
        return num_methods + 1  # Diff_Moving, Diff_Method1, ..., Diff_MethodM
    elif category == 'deformation':
        return num_methods      # Method1_field, ..., MethodM_field
    else:
        print(f"Warning: Unknown category '{category}'. Assuming 1 data subplot.")
        return 1


def get_slice_obj(np_array, axis, idx_slice):
    """
    Helper function for dynamic slicing

    e.g.
        np_array.shape = (H, W, D)
        axis = 0
        idx_slice = xx_slice
        slice_obj = get_slice_obj(np_array, axis, idx_slice)
        np_array[slice_obj] # is a 2D array of shape (W, D)
    """
    s = [slice(None)] * np_array.ndim
    s[axis] = idx_slice
    return tuple(s)

def plot_registration_results(
    data_rows,
    num_methods,
    row_order=None, # Optional list of indices to reorder data_rows
    # --- Title ---
    suptitle_text=None,
    col_titles=None,
    # --- Figure Layout & Sizing ---
    dpi=150,
    subplot_width=3.0,  # inches for each data subplot
    subplot_aspect_ratio=1.0, # height = width * ratio
    # --- Spacing (factors of subplot_width/height or absolute) ---
    wspace_factor=0.05, 
    hspace_factor=0.05, 
    right_margin_factor=0.95, 
    top_margin_factor=0.90,
    # --- Font Sizes ---
    suptitle_fontsize=16,
    col_title_fontsize=14,
    row_title_fontsize=14,
    tick_label_fontsize=14, # Also base for colorbar label    
    # --- Output ---
    output_filename=None, # Optional: if provided, save the figure
    show_figure=True,
):
    """
    Main function to plot registration results.
        Generates a highly customizable grid plot for image registration results.
        Shared colorbars for 'diff' and 'deformation' categories are placed in empty
        conceptual slots within the main data grid, using make_axes_locatable for precise sizing.
    """
    if not data_rows:
        print("Warning: No data_rows provided. Nothing to plot.")
        return

    # --- 1. Process row_order ---
    if row_order:
        try:
            processed_data_rows = [data_rows[i] for i in row_order]
        except IndexError:
            print("Warning: Invalid row_order indices. Using original order.")
            processed_data_rows = list(data_rows) 
    else:
        processed_data_rows = list(data_rows) 

    actual_num_rows = len(processed_data_rows)
    if actual_num_rows == 0:
        print("Warning: No rows to plot after processing row_order.")
        return

    # --- 2. Pre-scan for GridSpec layout requirements ---
    num_conceptual_data_columns = num_methods + 2
    max_data_cols_in_any_row = num_conceptual_data_columns 

    has_any_row_title = False
    for row_config in processed_data_rows: 
        if row_config.get('row_title'):
            has_any_row_title = True
    
    # --- 3. Setup GridSpec dimensions and width ratios ---
    gs_num_cols = max_data_cols_in_any_row 
    width_ratios = [1] * max_data_cols_in_any_row 

    row_title_col_idx_in_gs = -1 
    row_title_col_width_ratio = 0.15 
    if has_any_row_title:
        row_title_col_idx_in_gs = gs_num_cols 
        gs_num_cols += 1
        width_ratios.append(row_title_col_width_ratio)

    # --- 4. Calculate Figure size ---
    subplot_height = subplot_width * subplot_aspect_ratio
    
    current_total_width_ratio = sum(width_ratios[:max_data_cols_in_any_row])
    if has_any_row_title and row_title_col_idx_in_gs != -1 and row_title_col_idx_in_gs < len(width_ratios):
         current_total_width_ratio += width_ratios[row_title_col_idx_in_gs]
    
    fig_width_inches = current_total_width_ratio * subplot_width
        
    if gs_num_cols > 1 : 
        fig_width_inches += (gs_num_cols -1) * wspace_factor * subplot_width

    fig_height_inches = actual_num_rows * subplot_height
    if actual_num_rows > 1: 
         fig_height_inches += (actual_num_rows -1) * hspace_factor * subplot_height
    
    fig_width_inches *= 1.05 
    fig_height_inches *= 1.1 

    # --- 5. Create Figure and GridSpec ---
    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=dpi)
    gs = gridspec.GridSpec(actual_num_rows, gs_num_cols,
                           width_ratios=width_ratios,
                           height_ratios=[1] * actual_num_rows, 
                           wspace=wspace_factor, hspace=hspace_factor)

    # --- 6. Iterate through rows and plot ---
    for plot_row_idx, row_config in enumerate(processed_data_rows):
        category = row_config.get('category', 'unknown')
        data_list = row_config.get('data', [])
        disp_settings_list = row_config.get('disp_settings', [])
        
        num_data_items_in_row = _get_num_data_subplots(category, num_methods)

        if len(data_list) != num_data_items_in_row:
            print(f"Warning: Row {plot_row_idx} ('{row_config.get('content_description', 'N/A')}'): "
                  f"Expected {num_data_items_in_row} data arrays for category '{category}', "
                  f"but got {len(data_list)}. Plotting available items.")
            num_data_items_in_row = min(len(data_list), num_data_items_in_row) 

        start_col_offset_this_row = 0
        if category == 'diff':
            start_col_offset_this_row = 1  
        elif category == 'deformation':
            start_col_offset_this_row = 2  

        mappable_for_shared_cb = None 
        
        # --- Plot actual data images for the row ---
        for img_data_idx in range(num_data_items_in_row): 
            actual_gs_col_idx = start_col_offset_this_row + img_data_idx
            
            if actual_gs_col_idx >= max_data_cols_in_any_row: 
                print(f"Warning: Row {plot_row_idx}, Img Idx {img_data_idx}: GS col {actual_gs_col_idx} "
                      f"out of bounds for data cols ({max_data_cols_in_any_row}). Skipping.")
                continue

            ax = fig.add_subplot(gs[plot_row_idx, actual_gs_col_idx])
            img_data = data_list[img_data_idx] 

            current_disp_setting = _get_default_disp_setting() 
            user_setting_source = None
            if img_data_idx < len(disp_settings_list): 
                user_setting_source = disp_settings_list[img_data_idx]
            elif disp_settings_list: 
                user_setting_source = disp_settings_list[0]
            
            if user_setting_source: 
                for key in current_disp_setting:
                    if key in user_setting_source:
                        if isinstance(current_disp_setting[key], dict) and isinstance(user_setting_source[key], dict):
                            current_disp_setting[key].update(user_setting_source[key])
                        else:
                            current_disp_setting[key] = user_setting_source[key] 
            
            plot_type = current_disp_setting.get('plot_type', 'imshow')
            im_obj = None 
            
            if plot_type == 'imshow':
                params = current_disp_setting['imshow_params']
                im_obj = ax.imshow(img_data, **params)
            elif plot_type == 'grid': 
                params = current_disp_setting['grid_params']
                ax.set_facecolor(params['background_color'])
                if isinstance(img_data, np.ndarray) and img_data.ndim == 2 and img_data.shape[0] > 0 and img_data.shape[1] > 0:
                     ax.imshow(np.zeros_like(img_data[:,:,0] if img_data.ndim==3 else img_data), 
                               cmap='gray', vmin=0, vmax=1, 
                               aspect=current_disp_setting['imshow_params']['aspect']) 
                
                num_lines = params['density']
                data_h, data_w = (img_data.shape[0], img_data.shape[1]) if hasattr(img_data, 'shape') and len(img_data.shape) >= 2 else (100,100) 
                
                xlims = (0, data_w)
                ylims = (data_h, 0) 
                
                for i in np.linspace(xlims[0], xlims[1], num_lines + 1, endpoint=True)[1:-1]: 
                    ax.axvline(i - 0.5, color=params['line_color'], lw=params['line_width']) 
                for i in np.linspace(ylims[0], ylims[1], num_lines + 1, endpoint=True)[1:-1]:
                    ax.axhline(i - 0.5, color=params['line_color'], lw=params['line_width'])
                ax.set_xlim(-0.5, data_w - 0.5)
                ax.set_ylim(data_h - 0.5, -0.5)

            elif plot_type == 'overlay':
                if isinstance(img_data, (list, tuple)) and len(img_data) == 2:
                    base_img, overlay_img = img_data
                    base_p = current_disp_setting['imshow_params'] 
                    overlay_p = current_disp_setting['overlay_params']
                    
                    ax.imshow(base_img, cmap=overlay_p.get('base_cmap', base_p['cmap']), 
                              vmin=base_p['vmin'], vmax=base_p['vmax'],
                              aspect=base_p['aspect'], interpolation=base_p['interpolation'])
                    im_obj = ax.imshow(overlay_img, cmap=overlay_p['overlay_cmap'], alpha=overlay_p['overlay_alpha'],
                                       vmin=overlay_p.get('vmin'), vmax=overlay_p.get('vmax'), 
                                       aspect=base_p['aspect'], interpolation=base_p['interpolation'])
                else:
                    ax.text(0.5, 0.5, "Invalid overlay data\n(expected list/tuple of 2)", ha='center', va='center', fontsize=tick_label_fontsize-2)
            elif plot_type == 'custom_plot_func' and callable(current_disp_setting.get('custom_plot_func')):
                try:
                    current_disp_setting['custom_plot_func'](ax, img_data, **current_disp_setting)
                except Exception as e:
                    print(f"Error in custom_plot_func for row {plot_row_idx}, img {img_data_idx}: {e}")
                    ax.text(0.5, 0.5, "Custom Plot Error", ha='center', va='center', fontsize=tick_label_fontsize-2, color='red')
            else:
                ax.text(0.5,0.5, f"Plot type\n'{plot_type}'\nnot implemented", ha='center', va='center', fontsize=tick_label_fontsize-2)

            # --- Contour Overlay ---
            contour_settings = current_disp_setting.get('contour_overlay', {})
            if contour_settings.get('show', False) and img_data is not None:
                try:
                    contour_level = contour_settings.get('level', 0.0)
                    contour_color = contour_settings.get('color', 'black')
                    contour_linewidth = contour_settings.get('linewidth', 0.8)
                    contour_linestyle = contour_settings.get('linestyle', '-')
                    
                    # Match origin with imshow if specified, otherwise default for contour is 'upper'
                    imshow_origin = current_disp_setting.get('imshow_params', {}).get('origin', 'upper')

                    ax.contour(img_data, 
                               levels=[contour_level], 
                               colors=[contour_color], # ax.contour expects a list of colors
                               linewidths=contour_linewidth,
                               linestyles=contour_linestyle,
                               origin=imshow_origin) 
                except Exception as e:
                    print(f"Error drawing contour for row {plot_row_idx}, img {img_data_idx}: {e}")
            # --- End Contour Overlay ---

            if im_obj and mappable_for_shared_cb is None: 
                 mappable_for_shared_cb = im_obj

            if not current_disp_setting.get('show_axis', False):
                ax.set_axis_off()
            else:
                ax.tick_params(labelsize=tick_label_fontsize)

            subplot_title_text = None
            row_override_titles = row_config.get('column_titles_override')
            if row_override_titles:
                if img_data_idx < len(row_override_titles) and row_override_titles[img_data_idx] is not None:
                    subplot_title_text = row_override_titles[img_data_idx]
            elif plot_row_idx == 0 and col_titles: 
                if actual_gs_col_idx < len(col_titles) and col_titles[actual_gs_col_idx] is not None:
                    subplot_title_text = col_titles[actual_gs_col_idx]
            
            if subplot_title_text:
                ax.set_title(subplot_title_text, fontsize=col_title_fontsize)

            cb_ind_settings = current_disp_setting.get('colorbar', {})
            row_shared_cb_settings = row_config.get('shared_row_colorbar', {}) 
            if cb_ind_settings.get('show', False) and \
               not row_shared_cb_settings.get('show', False) and \
               im_obj: 
                try:
                    divider = make_axes_locatable(ax)
                    orient = cb_ind_settings.get('orientation', 'vertical')
                    side = "right" if orient == 'vertical' else "bottom"
                    size_str = "5%" if orient == 'vertical' else "10%" 
                    
                    cax_ind = divider.append_axes(side, size=size_str, pad=0.05) 
                    cb = fig.colorbar(im_obj, cax=cax_ind, orientation=orient)
                    cb.ax.tick_params(labelsize=tick_label_fontsize)
                    if cb_ind_settings.get('label'):
                        cb.set_label(cb_ind_settings.get('label'), fontsize=tick_label_fontsize)
                except Exception as e:
                    print(f"Error creating individual colorbar for subplot: {e}")

        # --- Shared Row Colorbar (after all subplots in the row are drawn) ---
        shared_cb_settings = row_config.get('shared_row_colorbar', {})
        if shared_cb_settings.get('show', False) and mappable_for_shared_cb:
            cb = None 
            ax_cell_for_colorbar = None # The GridSpec cell that will host the colorbar's thin axes
            cax_final_for_colorbar = None # The thin axes appended for the colorbar itself

            target_gs_col_for_cb = -1
            if category == 'diff':
                target_gs_col_for_cb = 0 # Conceptual "Fixed" column
            elif category == 'deformation':
                target_gs_col_for_cb = 1 # Conceptual "Moving" column
            
            if target_gs_col_for_cb != -1:
                # Get the GridSpec cell where the colorbar will be placed
                ax_cell_for_colorbar = fig.add_subplot(gs[plot_row_idx, target_gs_col_for_cb])
                ax_cell_for_colorbar.set_axis_off() # Make this main cell invisible

                divider = make_axes_locatable(ax_cell_for_colorbar)
                
                try:
                    cb_orientation = shared_cb_settings.get('orientation', 'vertical')
                    cb_label_fontsize = max(5, tick_label_fontsize - 1)
                    ticks_on_left = False
                    label_pad_val = 10 
                    cb_thickness_percent_str = "15%" # How thick the colorbar should be (e.g., 15% of cell width)
                    cb_pad_from_cell_edge_str = "5%" # Padding from the main (invisible) part of the cell

                    if category == 'diff':
                        # Append a thin axes to the RIGHT side of the (invisible) ax_cell_for_colorbar (which is gs[...,0])
                        cax_final_for_colorbar = divider.append_axes(
                            "right",  # position of the new axes relative to the main part of ax_cell_for_colorbar
                            size=cb_thickness_percent_str, 
                            pad=cb_pad_from_cell_edge_str 
                        )
                        ticks_on_left = True # Ticks and label on the left of the colorbar
                        label_pad_val = 20 # Adjust padding for left-sided label
                    elif category == 'deformation':
                        # Append a thin axes to the RIGHT side of the (invisible) ax_cell_for_colorbar (which is gs[...,1])
                        cax_final_for_colorbar = divider.append_axes(
                            "right", # # position of the new axes relative to the main part of ax_cell_for_colorbar
                            size=cb_thickness_percent_str, 
                            pad=cb_pad_from_cell_edge_str
                        )
                        ticks_on_left = True # Ticks and label on the left of the colorbar
                        label_pad_val = 15
                    
                    if cax_final_for_colorbar:
                        # Draw the colorbar into the new, precisely sized cax_final_for_colorbar
                        cb = fig.colorbar(mappable_for_shared_cb, 
                                          cax=cax_final_for_colorbar, 
                                          orientation=cb_orientation)
                        
                        # --- Apply custom ticks if provided ---
                        norm_ticks = shared_cb_settings.get('tick_values_normalized')
                        actual_labels = shared_cb_settings.get('tick_labels_actual')
                        if norm_ticks and actual_labels and len(norm_ticks) == len(actual_labels):
                            cb.set_ticks(norm_ticks)
                            cb.set_ticklabels(actual_labels)
                        # --- End custom ticks ---
                        
                        if ticks_on_left:
                            cb.ax.yaxis.set_ticks_position('left')
                            cb.ax.yaxis.set_label_position('left')
                        else: 
                            cb.ax.yaxis.set_ticks_position('right')
                            cb.ax.yaxis.set_label_position('right')

                        cb.ax.tick_params(labelsize=tick_label_fontsize, **shared_cb_settings.get('tick_params', {}))
                        if shared_cb_settings.get('label'):
                            cb.set_label(shared_cb_settings.get('label'), 
                                         fontsize=cb_label_fontsize, 
                                         labelpad=label_pad_val)
                except Exception as e:
                    print(f"Error creating shared colorbar in-grid for row {plot_row_idx} (category '{category}'): {e}")
                    err_ax_to_mark = cax_final_for_colorbar if cax_final_for_colorbar else ax_cell_for_colorbar
                    if err_ax_to_mark and not hasattr(err_ax_to_mark, '_plot_error_marked'): 
                        try:
                            err_ax_to_mark.text(0.5,0.5, "CB Err", ha='center', va='center', fontsize=max(5, tick_label_fontsize-2), color='red')
                            if not cax_final_for_colorbar : err_ax_to_mark.set_axis_off() 
                            err_ax_to_mark._plot_error_marked = True 
                        except Exception: pass
            else: 
                print(f"Warning: Row {plot_row_idx} (category '{category}') requests shared colorbar, but no in-grid placement logic defined for this category.")


        # --- Row Title (after all subplots in the row are drawn) ---
        row_title_text = row_config.get('row_title')
        if row_title_text and row_title_col_idx_in_gs != -1: 
            rt_ax = None 
            try:
                rt_ax = fig.add_subplot(gs[plot_row_idx, row_title_col_idx_in_gs])
                rt_ax.text(0.5, 0.5, row_title_text, 
                           ha='center', va='center', rotation=270, 
                           fontsize=row_title_fontsize)
                rt_ax.set_axis_off()
            except Exception as e:
                print(f"Error creating row title for row {plot_row_idx}: {e}")
                if rt_ax and not hasattr(rt_ax, '_plot_error_marked'): 
                    try:
                        rt_ax.text(0.5,0.5, "RT Err", ha='center', va='center', fontsize=max(5,tick_label_fontsize-2), color='red')
                        rt_ax.set_axis_off()
                        rt_ax._plot_error_marked = True
                    except Exception: pass

    # --- 7. Suptitle and Final Layout Adjustments ---
    if suptitle_text:
        # fig.suptitle(suptitle_text, fontsize=suptitle_fontsize, y=top_margin_factor + 0.03 if top_margin_factor < 0.95 else 0.98)
        fig.suptitle(suptitle_text, fontsize=suptitle_fontsize, y=top_margin_factor + 0.05) # leave more space for the suptitle

    fig.subplots_adjust(
        left=0.05, bottom=0.05, 
        right=right_margin_factor, 
        top=top_margin_factor,     
        wspace=gs.wspace,          
        hspace=gs.hspace           
    )

    # --- 8. Save or Show ---
    if output_filename:
        try:
            plt.savefig(output_filename, dpi=dpi, bbox_inches='tight') 
            print(f"Figure saved to {output_filename}")
        except Exception as e:
            print(f"Error saving figure: {e}")
    
    if show_figure:
        plt.show()
    else:
        plt.close()

# --- Example Usage (Updated to better test alignment and spacing) ---
if __name__ == '__main__':
    num_example_methods = 2 

    fixed_img = np.random.rand(64, 64) * 0.8
    moving_img = np.random.rand(64, 64) * 0.7
    
    deformed_m_results = [np.random.rand(64, 64) for _ in range(num_example_methods)]
    diff_m_results = [(np.random.rand(64, 64) * 1.5 - 0.75) for _ in range(num_example_methods)] 
    deformation_m_results = [np.random.rand(64,64) for _ in range(num_example_methods)]


    example_data_rows = [
        { 
            'category': 'deformed', 
            'content_description': 'Deformed Images (T1w)', 
            'row_title': 'Deformed',
            'data': [fixed_img, moving_img] + deformed_m_results,
            'disp_settings': [{'plot_type': 'imshow', 'imshow_params': {'cmap': 'Greys'}}] * (num_example_methods + 2)
        },
        { 
            'category': 'diff', 
            'content_description': 'Difference Maps (Registered - Fixed)', 
            'row_title': 'Difference', 
            'data': [(moving_img - fixed_img)*2] + diff_m_results, 
            'disp_settings': [{'plot_type': 'imshow', 'imshow_params': {'cmap': 'coolwarm', 'vmin': -0.5, 'vmax': 0.5}}] * (num_example_methods + 1),
            'shared_row_colorbar': {
                'show': True, 
                'label': 'HU Difference',
                'tick_values_normalized': [-0.5, 0, 0.5], 
                'tick_labels_actual': ['-1000', '0', '1000'] 
            } 
        },
        { 
            'category': 'deformation', 
            'content_description': 'Deformation Magnitude', 
            'row_title': 'Deform Mag',
            'data': deformation_m_results,
            'disp_settings': [{'plot_type': 'imshow', 'imshow_params': {'cmap': 'viridis', 'vmin':0, 'vmax':1}}] * num_example_methods,
            'shared_row_colorbar': { # This will also use the in-grid placement
                'show': True, 
                'label': 'Magnitude Value',
                'tick_values_normalized': [0, 0.5, 1.0], 
                'tick_labels_actual': ['0.0', '0.5', '1.0'] 
            } 
        },
         { 
            'category': 'deformed', 
            'content_description': 'Deformed Labels (Segmentation)', 
            'row_title': 'Def. Label',
            'data': [np.random.randint(0,4, (64,64)), np.random.randint(0,4, (64,64))] + \
                    [np.random.randint(0,4, (64,64)) for _ in range(num_example_methods)],
            'disp_settings': [{'plot_type': 'imshow', 
                               'imshow_params': {'cmap': 'tab10', 'interpolation':'nearest', 'vmin':0, 'vmax':9}}] * \
                             (num_example_methods + 2)
        },
         { 
            'category': 'deformation', 
            'content_description': 'Deformation Grid Visualization', 
            'row_title': 'Def. Grid', 
            'data': [np.zeros((32,64)) for _ in range(num_example_methods)], 
            'disp_settings': [{'plot_type': 'grid', 
                               'grid_params':{'density':5, 'line_color':'lime', 'background_color':'#222222'}} 
                              for _ in range(num_example_methods)],
             'column_titles_override': [f"Grid M{i+1}" for i in range(num_example_methods)] 
        }
    ]

    global_col_titles = ["Fixed Target", "Moving Source"] + [f"Method {i+1}" for i in range(num_example_methods)]
    
    plot_registration_results(
        data_rows=example_data_rows,
        num_methods=num_example_methods,
        suptitle_text=f"Registration Overview ({num_example_methods} Methods)",
        col_titles=global_col_titles,
        subplot_width=2.2, 
        dpi=100,
        wspace_factor=0.1, # Increased from 0.05 for this example to give a bit more space
        hspace_factor=0.15, 
        top_margin_factor=0.92, 
        right_margin_factor=0.90 # Reduced to give more space on the right for titles
    )

