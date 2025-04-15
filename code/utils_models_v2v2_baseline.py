

"""
List of classes/functions:
    model_predict_v2
    get_models_v2    
        get_model_vxm
        get_model_tm
        get_model_sitreg
        # VFA is diectly imported as VFA_v0

List of varibles:
    model_category_info = [list_standard, list_standard_concat, list_sitreg]
    

"""

######### general imports
import sys

import torch
import torch.nn as nn
import torch.nn.functional as nnf

######### my imports (related to baselines)
### NOTE: might need to modify here for new models!!!
from networks.model_vfa_v0 import VFA_v0
# from networks.flow_estimator import DIRNet ### PENDING TEST, HOPE IT WORKS
from networks_v2.flow_estimator import DIRNet ### PENDING TEST, HOPE IT WORKS

from utils_models_v2v2_common import count_trainable_parameters, report_gpu_memory





#######################################################################################
### Global parameters for model category

list_standard = [
    'bs3_vfa_v0',
]
list_standard_concat = [
    # 'bs1_vxm_v0_1',  # not used currently
    'bs1_vxm_v0_2', 'bs1_vxm_v0_2_1t', 'bs1_vxm_v0_2_2s', 'bs1_vxm_v0_2_3m', 'bs1_vxm_v0_2_4l', # bs1_vxm_v0_2 and bs1_vxm_v0_2_1t are the same
    'bs2_tm_v0',
]
list_sitreg = [
    'bs4_sitreg_v0',
]

model_category_info = [list_standard, list_standard_concat, list_sitreg]
# call
# list_standard, list_standard_concat, list_sitreg = model_category_info
#######################################################################################

def model_predict_v2(moving, fixed, model, model_type, spatial_trans=None):
    
    if model_type in list_standard:
        deformed, dvf = model(moving, fixed)
        return deformed, dvf
    elif model_type in list_standard_concat:
        model_input = torch.cat((moving, fixed), dim=1)
        deformed, dvf = model(model_input)
        return deformed, dvf
    elif model_type in list_sitreg:
        ### single
        # ((forward_mapping, inverse_mapping),) = model(image_1=moving, image_2=fixed)
        # dvf = forward_mapping._data.displacements
        # deformed = spatial_trans(moving, dvf)
        image_1 = moving
        image_2 = fixed
        ((forward_mapping, inverse_mapping),) = model(image_1=image_1, image_2=image_2)
        dvf_12 = forward_mapping._data.displacements
        dvf_21 = inverse_mapping._data.displacements

        deformed_1 = spatial_trans(image_1, dvf_12)
        deformed_2 = spatial_trans(image_2, dvf_21)
        
        return deformed_1, deformed_2, dvf_12, dvf_21
        

def get_models_v2(
    img_size, model_type, path_model=None, device=None, INFO=True, 
    dir_vxm=None, dir_tm=None, return_info=False
):
    """
    dir_vxm = voxelmorph_models_path
    dir_tm = transmorph_dir
    
    Note:
        might be better to return list_standard, list_standard_concat
    """
    
    print(f"Initializing model_type: {model_type} from utils_models_v2v2_baseline.get_models_v2")
    
    
    ###### Define models
    ### Baseline VoxelMorph
    if model_type.startswith('bs1_vxm_v0'):
        model = get_model_vxm(img_size, model_type, dir_vxm)
        
    ### Baseline TransMorph
    elif model_type == 'bs2_tm_v0':
        model = get_model_tm(img_size, model_type, dir_tm)
        
    ### Baseline VFA (vector field attention)
    elif model_type == 'bs3_vfa_v0':
        unet_model = VFA_v0()
        model = DIRNet(img_size, unet_model)
        
    ### Baseline SITReg
    elif model_type == 'bs4_sitreg_v0':
        model = get_model_sitreg(img_size, model_type)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    
    
    ###### load model weights
    if path_model is not None:
        # model.load_state_dict(torch.load(path_model)["state_dict"])
        model.load_state_dict(torch.load(path_model, weights_only=False)["state_dict"])
        if INFO:
            print(f"Loading {model_type} weights from: {path_model}")
    else:
        if INFO:
            print(f"path_model is None: model weights not loaded")
    
    ###### move to device
    if device is not None:
        model.to(device)
    
    ###### print number of trainable parameters
    if INFO:
        print(f"Number of trainable parameters model: {count_trainable_parameters(model):,}")

    if return_info:
        return model, model_category_info
    else:
        return model


def get_model_vxm(img_size, model_type, dir_vxm):
    """
    https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/main/VoxelMorph/models.py
        bs1_vxm_v0_1:
            VxmDense_1
        bs1_vxm_v0_2:
            VxmDense_2
    """
    
    # Function to dynamically load a module from a file path
    import importlib.util
    def load_module(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Cannot find module {module_name} at {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    ### loading packages
    voxelmorph_models = load_module("voxelmorph_models", dir_vxm)
    VxmDense_1 = voxelmorph_models.VxmDense_1
    VxmDense_2 = voxelmorph_models.VxmDense_2

    ### 'bs1_vxm_v0_1'
    if model_type == 'bs1_vxm_v0_1':
        # default nb_unet_features = ((8, 32, 32, 32), (32, 32, 32, 32, 32, 8, 8))
        # https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/9b88e630398949bd6a429871fbc516e212334353/VoxelMorph/models.py#L113C27-L113C72
        model = VxmDense_1(img_size)

    ### 'bs1_vxm_v0_2', 'bs1_vxm_v0_2_1t', 'bs1_vxm_v0_2_2s', 'bs1_vxm_v0_2_3m', 'bs1_vxm_v0_2_4l', 
    elif model_type.startswith('bs1_vxm_v0_2'):
        if model_type == 'bs1_vxm_v0_2' or model_type == 'bs1_vxm_v0_2_1t':
            # default nb_unet_features = ((16, 32, 32, 32), (32, 32, 32, 32, 32, 16, 16))
            nb_unet_features = ((16, 32, 32, 32),   (32, 32, 32, 32, 32, 16, 16))
        elif model_type == 'bs1_vxm_v0_2_2s':
            nb_unet_features = ((16, 32, 64, 96),   (128, 128, 96, 64, 32, 32, 32))
        elif model_type == 'bs1_vxm_v0_2_3m':
            nb_unet_features = ((16, 32, 64, 128),  (256, 256, 128, 64, 32, 32, 32))
        elif model_type == 'bs1_vxm_v0_2_4l':
            ### nnU-Net size: https://catalog.ngc.nvidia.com/orgs/nvidia/resources/nnunet_for_pytorch
            ### also make the last 2 layers 64 ...
            nb_unet_features = ((32, 64, 128, 256), (320, 320, 256, 128, 64, 64, 64))
            
        model = VxmDense_2(img_size, nb_unet_features)
        
    return model
    

def get_model_tm(img_size, model_type, dir_tm):
    
    import sys
    sys.path.insert(0, dir_tm)
    from models.TransMorph import CONFIGS as CONFIGS_TM
    import models.TransMorph as TransMorph
    
    config_default = CONFIGS_TM['TransMorph']
    # config modify size
    config_new = config_default.copy_and_resolve_references()
    config_new.img_size = img_size # overwrite img_size
    print('config_default.img_size: ', config_default.img_size) # (160, 192, 224)
    print('config_new.img_size: ', config_new.img_size) # (160, 224, 192)
    model = TransMorph.TransMorph(config_new)
    
    return model

def get_model_sitreg(img_size, model_type):
    path_stireg = '/homebase/DL_projects/wavereg/SITReg/src'
    import sys
    sys.path.insert(0, path_stireg)
    from model.activation import ReLUFactory
    from model.normalizer import GroupNormalizerFactory
    from model.sitreg import SITReg
    from model.sitreg.feature_extractor import EncoderFeatureExtractor
    from deformation_inversion_layer.fixed_point_iteration import AndersonSolver, MaxElementWiseAbsStopCriterion, RelativeL2ErrorStopCriterion

    import json
    config_path = '/homebase/DL_projects/wavereg/SITReg/src/scripts/configs/sitreg/lumir/cc_grad_1.0_very_deep_heavy.json'
    with open(config_path, mode="r", encoding="utf-8") as f:
        config = json.load(f)

    forward_fixed_point_solver = AndersonSolver(
        stop_criterion=MaxElementWiseAbsStopCriterion(min_iterations=2, max_iterations=50, threshold=1e-2))
    backward_fixed_point_solver = AndersonSolver(
        stop_criterion=RelativeL2ErrorStopCriterion(min_iterations=2, max_iterations=50, threshold=1e-2))

    if model_type == 'bs4_sitreg_v0':
        expected_n_param = 15076110 # 15,076,110
        
        activation_factory = ReLUFactory()
        normalizer_factory = GroupNormalizerFactory(4)
        
        affine_transformation_type = None
        
        n_input_channels = config["application"]["config"]["model"]["n_input_channels"]
        n_features_per_resolution = config["application"]["config"]["model"]["n_features_per_resolution"]
        n_convolutions_per_resolution = config["application"]["config"]["model"]["n_feature_convolutions_per_resolution"] # no key as n_features_per_resolution
        # input_shape = config["application"]["config"]["model"]["input_shape"]
        input_shape = img_size # overwrite img_size
        
        n_transformation_convolutions_per_resolution = config["application"]["config"]["model"]["n_transformation_convolutions_per_resolution"]
        n_transformation_features_per_resolution = config["application"]["config"]["model"]["n_transformation_features_per_resolution"]
        max_control_point_multiplier = config["application"]["config"]["model"]["max_control_point_multiplier"]
        transformation_downsampling_factor = config["application"]["config"]["model"]["transformation_downsampling_factor"]
        voxel_size = config["application"]["config"]["model"]["voxel_size"]
    
    print(f"Expected # of parameters of {model_type} is  {expected_n_param:,}")
    feature_extractor = EncoderFeatureExtractor(
                n_input_channels=n_input_channels,
                activation_factory=activation_factory,
                n_features_per_resolution=n_features_per_resolution,
                n_convolutions_per_resolution=n_convolutions_per_resolution,
                input_shape=input_shape,
                normalizer_factory=normalizer_factory,
    )
    model = SITReg(
        feature_extractor=feature_extractor,
        n_transformation_convolutions_per_resolution=n_transformation_convolutions_per_resolution,
        n_transformation_features_per_resolution=n_transformation_features_per_resolution,
        max_control_point_multiplier=max_control_point_multiplier,
        affine_transformation_type=affine_transformation_type,
        input_voxel_size=voxel_size,
        input_shape=input_shape,
        transformation_downsampling_factor=transformation_downsampling_factor,
        forward_fixed_point_solver=forward_fixed_point_solver,
        backward_fixed_point_solver=backward_fixed_point_solver,
        activation_factory=activation_factory,
        normalizer_factory=normalizer_factory,
    )
    
    return model    




