

print("Running all imports from common_imports_all_v1.py.py")

########################################################################################
### import common packages
########################################################################################
import os
import sys
import glob
import time
import yaml

import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib

import re
from natsort import natsorted

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

### for fp16 training
from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler

########################################################################################
### import my packages
########################################################################################
my_src_dir = os.path.abspath(
    '../code'
)
sys.path.insert(0, my_src_dir)

from utils_basics import numpy_overview, torch_overview, numpy2torch, torch2numpy
import utils_io as utils_io
from utils_logging import Logger_all
import utils_warp as utils_warp
import utils_models_v2v2_common as utils_models_v2_common
import utils_models_v2v2_baseline as utils_models_v2_baseline
import utils_models_v2v2_fedd as utils_models_v2_fedd
import utils_models_pretext as utils_models_pretext
from utils_losses_ncc import NCC_gauss, NCC_vxm, NCC_vxm_fast, NCC_vfa, NCC_vfa_fast
from utils_losses_mind import MIND_loss
from utils_losses_dice import Dice_vxm
import utils_rand_seed as utils_rand_seed
from utils_checkpoints_my import save_checkpoint
import utils_eval as utils_eval

###### dataset dependent ######
dataset_name = os.environ.get("DATASET_NAME")
print(f"Name of the dataset: {dataset_name}")
if dataset_name == "OASIS_v1":
    from utils_datasets_my import reorient_LIA_to_RAS, myDataset_OASIS_v1_json, myDataset_OASIS_pretext_v1_json
    path_data_config = "../configs/config_dataset_OASIS_v1.yaml"
elif dataset_name == "LUMIR_v1":
    from utils_datasets_lumir import L2RLUMIRDataset, L2RLUMIRJSONDataset, L2RLUMIRJSONDataset_subset
    path_data_config = "../configs/config_dataset_LUMIR_v1.yaml"
elif dataset_name.startswith("L2R20Task3_AbdominalCT"):
    from utils_datasets_my import myDataset_L2R20TASK3CT_v1_json
    if dataset_name == "L2R20Task3_AbdominalCT_v1":
        path_data_config = "../configs/config_dataset_L2R20Task3_AbdominalCT_v1.yaml"
    elif dataset_name == "L2R20Task3_AbdominalCT_v2":
        path_data_config = "../configs/config_dataset_L2R20Task3_AbdominalCT_v2.yaml"
else:
    raise ValueError(f"Unrecognized dataset_name: {dataset_name}")

########################################################################################
### load data config file and extract
########################################################################################
data_config = utils_io.load_yaml(path_data_config)

# Assert that dataset_name matches the one in the config
assert dataset_name == data_config['dataset_name'], (
    f"Mismatch: dataset_name ({dataset_name}) does not match config dataset_name ({data_config['dataset_name']})."
)
# dataset_name = data_config['dataset_name']
img_size = data_config['img_size']
transmorph_dir = data_config['transmorph_dir']
voxelmorph_models_path = data_config['voxelmorph_models_path']
train_dir = data_config['train_dir']
val_dir = data_config['val_dir']
test_dir = data_config['test_dir']
json_path = data_config['json_path']

########################################################################################
### Import modules from TransMorph
########################################################################################
sys.path.insert(0, transmorph_dir) # Add TransMorph directory to sys.path
import losses as tm_losses # NCC_vxm(), Grad3d(penalty='l2'), DiceLoss()
import utils as tm_utils # register_model, dice_val_VOI


########################################################################################
### Print information related to imports
########################################################################################
print(f"img_size: {img_size}")
print(f"train_dir: {train_dir}")
print(f"val_dir: {val_dir}")
print(f"test_dir: {test_dir}")
print(f"json_path: {json_path}")
print(f"transmorph_dir: {transmorph_dir}")
print(f"voxelmorph_models_path: {voxelmorph_models_path}")
