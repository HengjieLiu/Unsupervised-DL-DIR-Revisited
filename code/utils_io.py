
import yaml
import numpy as np
import nibabel as nib

"""
Function list:
    load_yaml
    save_nii_lumir
    load_nifti_to_array
"""



def load_yaml(file_path):
    """
    Load YAML file from the given file path and return its content as a dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    
    
def save_nii_lumir(img, filename, pix_dim=[1., 1., 1.]):
    """
    Direct copy from https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/main/TransMorph/infer_TransMorphTVF.py
    """
    x_nib = nib.Nifti1Image(img, np.eye(4))
    x_nib.header.get_xyzt_units()
    x_nib.header['pixdim'][1:4] = pix_dim
    x_nib.to_filename('{}.nii.gz'.format(filename))
    
    
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