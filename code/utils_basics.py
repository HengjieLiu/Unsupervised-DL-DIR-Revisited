"""
General description:
    Basic util functions:
        for numpy arrays and pytorch tensors
        for list printing

Function/Class list:
    numpy_overview
    torch_overview
    numpy2torch
    torch2numpy
    rmse_numpy
    rmse_torch
    print_list
    print_and_compare_list
"""

import numpy as np
import torch


####################################################################################################
### PART 1: 
####################################################################################################

def numpy_overview(array: np.ndarray, varname: str = None) -> None:
    if varname is not None:
        print(f'{varname}: {array.dtype}, {array.ndim}D, shape={array.shape}, min={np.min(array)}, max={np.max(array)}')
    else:
        print(f'{array.dtype}, {array.ndim}D, shape={array.shape}, min={np.min(array)}, max={np.max(array)}')

        
def torch_overview(tensor: torch.Tensor, varname: str = None) -> None:
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


def rmse_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """Compute RMSE between two numpy arrays."""
    return np.sqrt(np.mean((x - y) ** 2))


def rmse_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute RMSE between two torch tensors."""
    return torch.sqrt(torch.mean((x - y) ** 2))



####################################################################################################
### PART 2: Printing functions
####################################################################################################

def print_list(mylist: list, print_idx: bool = True) -> None:
    """Print a list with optional index."""
    print(f"Length of list: {len(mylist)}")
    for idx, item in enumerate(mylist):
        if print_idx:
            print(f'{str(idx+1).zfill(2)}: {item}')
        else:
            print(item)


def print_and_compare_list(list_of_str_1: list, list_of_str_2: list, print_idx: bool = True) -> None:
    """
    Print two lists side by side for comparison, aligned by spaces.
    """
    assert len(list_of_str_1) == len(list_of_str_2), 'Input lists must have the same length.'
    
    max_len_1 = len(max(list_of_str_1, key=len))
    max_len_2 = len(max(list_of_str_2, key=len))

    print(f'Lengths: {len(list_of_str_1)} {len(list_of_str_2)}')
    
    for idx, (str_1, str_2) in enumerate(zip(list_of_str_1, list_of_str_2)):
        if print_idx:
            print(f'{str(idx+1).zfill(2)}: {str_1:<{max_len_1}} {str_2:<{max_len_2}}')
        else:
            print(f'{str_1:<{max_len_1}} {str_2:<{max_len_2}}')
