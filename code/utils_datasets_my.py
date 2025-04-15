
"""
List of functions and classes:
    def reorient_LIA_to_RAS

    class myDataset_OASIS_v1_json
    class myDataset_OASIS_v1_json_UnitTest
    class myDataset_OASIS_pretext_v1_json

    class myDataset_L2R20TASK3CT_v1_json
    class myDataset_L2R20TASK3CT_v1_json_UnitTest
    
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib


def reorient_LIA_to_RAS(arr):
    """
    Reorients an array from LIA+ to RAS+ (i.e. converts from OASIS to LUMIR convention).
    
    Process:
      - LIA+ -> LAI+   via transpose (swap 2nd and 3rd dimensions)
      - LAI+ -> RAI+   via flipping along axis 0
      - RAI+ -> RAS+   via flipping along axis 2
    """
    arr = np.transpose(arr, [0, 2, 1])
    arr = np.flip(arr, 0)
    arr = np.flip(arr, 2)
    return arr


class myDataset_OASIS_v1_json(Dataset):
    """
    Dataset class for loading OASIS MRI images (and segmentation maps) using a JSON file.
    
    JSON structure:
      - "training": list of {"image": "./train/imgXXXX.nii.gz", "seg": "./train/segXXXX.nii.gz"}
      - "validation" and "test_XXX": list of {
             "fixed": "./val/imgXXXX.nii.gz" or "./test/imgXXXX.nii.gz",
             "fixed_seg": "./val/segXXXX.nii.gz" or "./test/segXXXX.nii.gz",
             "moving": "./val/imgYYYY.nii.gz" or "./test/imgYYYY.nii.gz",
             "moving_seg": "./val/segYYYY.nii.gz" or "./test/segYYYY.nii.gz"
          }
    
    For stage "train", the loader randomly pairs two training images.
    For "validation" and test stages, it uses the pre-defined fixed/moving pairs.
    
    An optional preprocessing step is applied to both images and segmentation maps if
    reorient_to_RAS is set to True (default True).
    
    Returns:
        with_seg=True:
            if with_id:
                moving, fixed, moving_seg, fixed_seg, moving_id, fixed_id = (x, y, x_seg, y_seg, x_id, y_id)
            else:
                moving, fixed, moving_seg, fixed_seg = (x, y, x_seg, y_seg)
        with_seg=False:
            if with_id:
                moving, fixed, moving_id, fixed_id = (x, y, x_id, y_id)
            else:
                moving, fixed = (x, y)
        
        Naming convention: 
            x: moving
            y: fixed
            returns (x, y) is in (moving, fixed) order!!!
            
    """
    def __init__(self, 
                 base_dir, json_path, stage='train', subset_size=None, 
                 with_seg=False, with_id=False, reorient_to_RAS=True, 
                 INFO=True, DEBUG=False):
        
        with open(json_path, 'r') as f:
            d = json.load(f)
        self.base_dir = base_dir  # Base directory to prepend to relative file paths.
        self.stage = stage.lower()
        self.with_seg = with_seg
        self.with_id = with_id
        self.reorient_to_RAS = reorient_to_RAS
        self.INFO = INFO
        self.DEBUG = DEBUG
        
        if self.stage == 'train':
            if subset_size is None:
                self.items = d['training']
            else:
                self.items = random.sample(d['training'], subset_size) # sample subset_size from all items

            if self.INFO:
                # Extract the image identifiers from the file names (e.g., "img0005.nii.gz" → "0005")
                image_ids = [os.path.basename(item['image']).replace('img', '').split('.')[0] for item in self.items]
                image_ids = sorted(image_ids)  # sort them
                print(f"{len(image_ids)} images are chosen for training")
                print("List of images: " + ", ".join(image_ids))

        elif self.stage == 'validation' or self.stage.startswith('test'):
            self.items = d[self.stage]
        else:
            raise ValueError(f"Stage '{stage}' not recognized. Choose from 'train', 'validation', or 'test_*'.")

    def __len__(self):
        return len(self.items)
    
    ### From  https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/OASIS/TransMorph/data/datasets.py#L10 
    # (not used for now)
    # def one_hot(self, img, C):
    #     out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
    #     for i in range(C):
    #         out[i,...] = img == i
    #     return out
    
    def __getitem__(self, index):
        if self.stage == 'train':
            # For training: randomly pair one training image with another.
            ### NOTE:  Previously I was worried that this pairing strategy would not work
            #              if I break the training loop early (e.g. 100 steps),
            #              as I am affraid the moving image would be only drawn from 
            #              the first 100 images.
            #              But it turns out to be OK, since we use shuffle = True when defining dataloader
            #              train_loader = DataLoader(train_set, batch_size=1, shuffle=shuffle)            
            mov_dict = self.items[index]
            fixed_candidates = self.items.copy()
            del fixed_candidates[index]
            random.shuffle(fixed_candidates)
            fix_dict = fixed_candidates[0]
            
            # if self.DEBUG:
            #     print(len(self.items)) # 300
            #     print(self.items) # a sorted list of 300 dict: {'image': './train/img0001.nii.gz', 'seg': './train/seg0001.nii.gz'}
            
            mov_path = os.path.join(self.base_dir, mov_dict['image'])
            fix_path = os.path.join(self.base_dir, fix_dict['image'])
            x = nib.load(mov_path).get_fdata()
            y = nib.load(fix_path).get_fdata()

            # Extract IDs if requested (e.g., "./train/img0005.nii.gz" -> "0005")
            if self.with_id:
                x_id = os.path.basename(mov_dict['image']).replace('img', '').split('.')[0]
                y_id = os.path.basename(fix_dict['image']).replace('img', '').split('.')[0]
                
            if self.reorient_to_RAS:
                x = reorient_LIA_to_RAS(x)
                y = reorient_LIA_to_RAS(y)
            
            if self.with_seg:
                mov_seg_path = os.path.join(self.base_dir, mov_dict['seg'])
                fix_seg_path = os.path.join(self.base_dir, fix_dict['seg'])
                x_seg = nib.load(mov_seg_path).get_fdata()
                y_seg = nib.load(fix_seg_path).get_fdata()
                if self.reorient_to_RAS:
                    x_seg = reorient_LIA_to_RAS(x_seg)
                    y_seg = reorient_LIA_to_RAS(y_seg)
            
        else:
            # For validation and test stages: use the pre-defined pair.
            pair = self.items[index]
            mov_path = os.path.join(self.base_dir, pair['moving'])
            fix_path = os.path.join(self.base_dir, pair['fixed'])
            x = nib.load(mov_path).get_fdata()
            y = nib.load(fix_path).get_fdata()

            # Extract IDs if requested (e.g., "./train/img0005.nii.gz" -> "0005")
            if self.with_id:
                x_id = os.path.basename(pair['moving']).replace('img', '').split('.')[0]
                y_id = os.path.basename(pair['fixed']).replace('img', '').split('.')[0]
            
            if self.reorient_to_RAS:
                x = reorient_LIA_to_RAS(x)
                y = reorient_LIA_to_RAS(y)
            
            if self.with_seg:
                mov_seg_path = os.path.join(self.base_dir, pair['moving_seg'])
                fix_seg_path = os.path.join(self.base_dir, pair['fixed_seg'])
                x_seg = nib.load(mov_seg_path).get_fdata()
                y_seg = nib.load(fix_seg_path).get_fdata()
                if self.reorient_to_RAS:
                    x_seg = reorient_LIA_to_RAS(x_seg)
                    y_seg = reorient_LIA_to_RAS(y_seg)
            
        if self.DEBUG:
            if self.with_seg:
                print(f'mov_path: {mov_path}, mov_seg_path: {mov_seg_path}')
                print(f'fix_path: {fix_path}, fix_seg_path: {fix_seg_path}')
            else:
                print(f'mov_path: {mov_path}')
                print(f'fix_path: {fix_path}')
        
        # Add channel dimension and convert to torch tensors.
        x = x[None, ...]
        y = y[None, ...]
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        
        if self.with_seg:
            x_seg = x_seg[None, ...]
            y_seg = y_seg[None, ...]
            x_seg = torch.from_numpy(np.ascontiguousarray(x_seg)).float()
            y_seg = torch.from_numpy(np.ascontiguousarray(y_seg)).float()
            if self.with_id:
                return x, y, x_seg, y_seg, x_id, y_id # note str id with be returned in a list
            else:
                return x, y, x_seg, y_seg
        else:
            if self.with_id:
                return x, y, x_id, y_id # note str id with be returned in a list
            else:
                return x, y


        
class myDataset_OASIS_v1_json_UnitTest(myDataset_OASIS_v1_json):
    def __getitem__(self, index):
        """
        Instead of loading image data, return the 4-digit ID strings extracted
        from the file names. For training stage, the moving image is taken from
        self.items[index] and the fixed image is chosen at random from the rest.
        """
        if self.stage == 'train':
            # Get the moving item (dict)
            mov_dict = self.items[index]
            # Create a copy for fixed candidates and remove the moving one.
            fixed_candidates = self.items.copy()
            del fixed_candidates[index]
            random.shuffle(fixed_candidates)
            fix_dict = fixed_candidates[0]
            # Extract IDs from the file names (e.g., "./train/img0005.nii.gz" -> "0005")
            mov_id = os.path.basename(mov_dict['image']).replace('img','').split('.')[0]
            fix_id = os.path.basename(fix_dict['image']).replace('img','').split('.')[0]
            return mov_id, fix_id
        else:
            pair = self.items[index]
            mov_id = os.path.basename(pair['moving']).replace('img','').split('.')[0]
            fix_id = os.path.basename(pair['fixed']).replace('img','').split('.')[0]
            return mov_id, fix_id


            
class myDataset_OASIS_pretext_v1_json(Dataset):
    """
    Dataset class for loading OASIS MRI images (and segmentation maps) for pretext task: segmentation or autoencoder tasks.

    JSON structure:
      - "training": list of {"image": "./train/imgXXXX.nii.gz", "seg": "./train/segXXXX.nii.gz"}
      - "validation" and "test_*": list of {
             "fixed": "./val/imgXXXX.nii.gz" (or "./test/imgXXXX.nii.gz"),
             "fixed_seg": "./val/segXXXX.nii.gz" (or "./test/segXXXX.nii.gz"),
             "moving": "./val/imgYYYY.nii.gz" (or "./test/imgYYYY.nii.gz"),
             "moving_seg": "./val/segYYYY.nii.gz" (or "./test/segYYYY.nii.gz")
          }

    For segmentation/autoencoder tasks:
      - Training: load a single image (and segmentation) from the training list.
      - Validation/Test: flatten the pair structure so that both the fixed and moving images
        become independent samples. Thus, each pair produces two items.

    Returns:
      with_seg=True:
          (img, seg)
      with_seg=False:
          (img,)

    For validation/test, the length of the dataset is doubled.
    """
    def __init__(self, base_dir, json_path, stage='train', with_seg=False, reorient_to_RAS=True, DEBUG=False):
        with open(json_path, 'r') as f:
            d = json.load(f)
        self.base_dir = base_dir
        self.stage = stage.lower()
        self.with_seg = with_seg
        self.reorient_to_RAS = reorient_to_RAS
        self.DEBUG = DEBUG

        if self.stage == 'train':
            self.items = d['training']
        elif self.stage == 'validation' or self.stage.startswith('test'):
            # For validation/test, flatten each pair into two separate samples.
            pairs = d[self.stage]
            self.items = []
            for pair in pairs:
                # Add the fixed image as one sample.
                fixed_item = {"image": pair["fixed"]}
                if self.with_seg:
                    fixed_item["seg"] = pair["fixed_seg"]
                self.items.append(fixed_item)
                # Add the moving image as one sample.
                moving_item = {"image": pair["moving"]}
                if self.with_seg:
                    moving_item["seg"] = pair["moving_seg"]
                self.items.append(moving_item)
        else:
            raise ValueError(f"Stage '{stage}' not recognized. Choose from 'train', 'validation', or 'test_*'.")

    def __getitem__(self, index):
        item = self.items[index]
        img_path = os.path.join(self.base_dir, item["image"])
        img = nib.load(img_path).get_fdata()
        if self.reorient_to_RAS:
            img = reorient_LIA_to_RAS(img)
        
        if self.with_seg:
            seg_path = os.path.join(self.base_dir, item["seg"])
            seg = nib.load(seg_path).get_fdata()
            if self.reorient_to_RAS:
                seg = reorient_LIA_to_RAS(seg)
        
        if self.DEBUG:
            if self.with_seg:
                print(f"Image: {img_path}, Segmentation: {seg_path}")
            else:
                print(f"Image: {img_path}")

        # Add channel dimension and convert to torch tensor.
        img = img[None, ...]
        img = torch.from_numpy(np.ascontiguousarray(img)).float()
        if self.with_seg:
            seg = seg[None, ...]
            seg = torch.from_numpy(np.ascontiguousarray(seg)).float()
            return img, seg
        else:
            return img

    def __len__(self):
        return len(self.items)



class myDataset_L2R20TASK3CT_v1_json(Dataset):
    """
    Dataset class for loading CT images (and segmentation maps) using a JSON file.
    
    JSON structure:
      - "training": list of {"image": "./img/imgXXXX.nii.gz", "seg": "./label/labelXXXX.nii.gz"}
      - "validation": list of {
             "fixed": "./img/imgXXXX.nii.gz",
             "fixed_seg": "./label/labelXXXX.nii.gz",
             "moving": "./img/imgYYYY.nii.gz",
             "moving_seg": "./label/labelYYYY.nii.gz"
          }
    
    For stage "train", the loader randomly pairs two training images.
    For "validation", it uses the pre-defined fixed/moving pairs.
    
    By default, CT images are clipped between clip_min and clip_max (default -1000 and 1000 HU)
    and normalized to the range [0,1].
    
    Returns:
        With segmentation:
            if with_id:
                moving, fixed, moving_seg, fixed_seg, moving_id, fixed_id = (x, y, x_seg, y_seg, x_id, y_id)
            else:
                moving, fixed, moving_seg, fixed_seg = (x, y, x_seg, y_seg)
        Without segmentation:
            if with_id:
                moving, fixed, moving_id, fixed_id = (x, y, x_id, y_id)
            else:
                moving, fixed = (x, y)
                
    Note: The original reorientation steps have been removed.
    """
    def __init__(self, base_dir, json_path, stage='train', subset_size=None, 
                 with_seg=False, with_id=False, clip_min=-1000, clip_max=1000, INFO=True, DEBUG=False):
        
        with open(json_path, 'r') as f:
            d = json.load(f)
            
        self.base_dir = base_dir  # Base directory to prepend to relative file paths.
        self.stage = stage.lower()
        self.with_seg = with_seg
        self.with_id = with_id
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.INFO = INFO
        self.DEBUG = DEBUG
        
        if self.INFO:
            print(f"Clipping CT images between {self.clip_min} and {self.clip_max} HU and normalizing to 0-1.")
        
        if self.stage == 'train':
            if subset_size is None:
                self.items = d['training']
            else:
                self.items = random.sample(d['training'], subset_size)
            
            if self.INFO:
                image_ids = [os.path.basename(item['image']).replace('img', '').split('.')[0] for item in self.items]
                image_ids = sorted(image_ids)
                print(f"{len(image_ids)} images are chosen for training")
                print("List of images: " + ", ".join(image_ids))
                
        elif self.stage == 'validation':
            self.items = d['validation']
        elif self.stage == 'test':
            self.items = d['test']
        else:
            raise ValueError("Stage not recognized. Choose from 'train/validation/test'.")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        if self.stage == 'train':
            # For training: randomly pair one training image with another.
            mov_dict = self.items[index]
            fixed_candidates = self.items.copy()
            del fixed_candidates[index]
            random.shuffle(fixed_candidates)
            fix_dict = fixed_candidates[0]
            
            mov_path = os.path.join(self.base_dir, mov_dict['image'])
            fix_path = os.path.join(self.base_dir, fix_dict['image'])
            x = nib.load(mov_path).get_fdata()
            y = nib.load(fix_path).get_fdata()
            
            # Apply clipping and normalization to CT images.
            x = np.clip(x, self.clip_min, self.clip_max)
            x = (x - self.clip_min) / (self.clip_max - self.clip_min)
            y = np.clip(y, self.clip_min, self.clip_max)
            y = (y - self.clip_min) / (self.clip_max - self.clip_min)
            
            if self.with_id:
                x_id = os.path.basename(mov_dict['image']).replace('img', '').split('.')[0]
                y_id = os.path.basename(fix_dict['image']).replace('img', '').split('.')[0]
            
            if self.with_seg:
                mov_seg_path = os.path.join(self.base_dir, mov_dict['seg'])
                fix_seg_path = os.path.join(self.base_dir, fix_dict['seg'])
                x_seg = nib.load(mov_seg_path).get_fdata()
                y_seg = nib.load(fix_seg_path).get_fdata()
                # Note: segmentation maps are not normalized.
                
        else:  # Validation stage using pre-defined pairs.
            pair = self.items[index]
            mov_path = os.path.join(self.base_dir, pair['moving'])
            fix_path = os.path.join(self.base_dir, pair['fixed'])
            x = nib.load(mov_path).get_fdata()
            y = nib.load(fix_path).get_fdata()
            
            # Apply clipping and normalization to CT images.
            x = np.clip(x, self.clip_min, self.clip_max)
            x = (x - self.clip_min) / (self.clip_max - self.clip_min)
            y = np.clip(y, self.clip_min, self.clip_max)
            y = (y - self.clip_min) / (self.clip_max - self.clip_min)
            
            if self.with_id:
                x_id = os.path.basename(pair['moving']).replace('img', '').split('.')[0]
                y_id = os.path.basename(pair['fixed']).replace('img', '').split('.')[0]
            
            if self.with_seg:
                mov_seg_path = os.path.join(self.base_dir, pair['moving_seg'])
                fix_seg_path = os.path.join(self.base_dir, pair['fixed_seg'])
                x_seg = nib.load(mov_seg_path).get_fdata()
                y_seg = nib.load(fix_seg_path).get_fdata()
                # Note: segmentation maps are left unchanged.
        
        if self.DEBUG:
            if self.with_seg:
                print(f'mov_path: {mov_path}, mov_seg_path: {mov_seg_path}')
                print(f'fix_path: {fix_path}, fix_seg_path: {fix_seg_path}')
            else:
                print(f'mov_path: {mov_path}')
                print(f'fix_path: {fix_path}')
        
        # Add channel dimension and convert to torch tensors.
        x = x[None, ...]
        y = y[None, ...]
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        
        if self.with_seg:
            x_seg = x_seg[None, ...]
            y_seg = y_seg[None, ...]
            x_seg = torch.from_numpy(np.ascontiguousarray(x_seg)).float()
            y_seg = torch.from_numpy(np.ascontiguousarray(y_seg)).float()
            if self.with_id:
                return x, y, x_seg, y_seg, x_id, y_id
            else:
                return x, y, x_seg, y_seg
        else:
            if self.with_id:
                return x, y, x_id, y_id
            else:
                return x, y


class myDataset_L2R20TASK3CT_v1_json_UnitTest(myDataset_L2R20TASK3CT_v1_json):
    def __getitem__(self, index):
        """
        Instead of loading image data, return the 4-digit ID strings extracted
        from the file names. For training stage, the moving image is taken from
        self.items[index] and the fixed image is chosen at random from the rest.
        """
        if self.stage == 'train':
            # For training: randomly pair one training image with another.
            mov_dict = self.items[index]
            fixed_candidates = self.items.copy()
            del fixed_candidates[index]
            random.shuffle(fixed_candidates)
            fix_dict = fixed_candidates[0]
            # Extract IDs from the image file names (e.g., "./img/img0005.nii.gz" -> "0005")
            mov_id = os.path.basename(mov_dict['image']).replace('img', '').split('.')[0]
            fix_id = os.path.basename(fix_dict['image']).replace('img', '').split('.')[0]
            return mov_id, fix_id
        else:
            # For validation: use the pre-defined moving/fixed pairs.
            pair = self.items[index]
            mov_id = os.path.basename(pair['moving']).replace('img', '').split('.')[0]
            fix_id = os.path.basename(pair['fixed']).replace('img', '').split('.')[0]
            return mov_id, fix_id
