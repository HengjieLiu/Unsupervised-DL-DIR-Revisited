"""
List of classes/functions:
    model_predict_v2
    get_models_v2


List of varibles:
    NA
    

"""

######### general imports
import os
import sys
import yaml
import glob

import torch
import torch.nn as nn
import torch.nn.functional as nnf

######### my imports
sys.path.insert(0, '/homebase/DL_projects/wavereg/code/networks_v2')
from feature_encoder import FeatureEncoder
from deformation_decoder import DeformationDecoder
from networks_v2.flow_estimator import DIRNet # may remove networks_v2

from utils_models_v2v2_common import count_trainable_parameters, report_gpu_memory



def model_predict_v2(moving, fixed, model, bidir=False, spatial_trans=None):

    if not bidir:
        deformed, dvf = model(moving, fixed)
        return deformed, dvf
    
    ### bidir models (NOTE THE ORDER OF THE OUTPUTS)
    else:

        outputs = model(moving, fixed)

        ###### NOTE THIS ORDER!!! [ORDER 1] ######
        # Match with model outputs
        if len(outputs) == 4:
            deformed_1, dvf_12, deformed_2, dvf_21 = outputs
        elif len(outputs) == 5:
            deformed_1, dvf_12, deformed_2, dvf_21, extra = outputs
        else:
            raise ValueError(f"Expected outputs to have length 4 or 5, but got {len(outputs)}")

        ###### NOTE THIS ORDER!!! [ORDER 2] ######
        # match with what DIRNet requires for bidir
        return deformed_1, deformed_2, dvf_12, dvf_21

        ### REF: utils_models_v2v2_baseline.model_predict_v2
        # image_1 = moving
        # image_2 = fixed
        # ((forward_mapping, inverse_mapping),) = model(image_1, image_2)
        # dvf_12 = forward_mapping._data.displacements
        # dvf_21 = inverse_mapping._data.displacements

        # deformed_1 = spatial_trans(image_1, dvf_12)
        # deformed_2 = spatial_trans(image_2, dvf_21)
        
        # return deformed_1, deformed_2, dvf_12, dvf_21
    


def get_models_v2(
    img_size, dir_config, model_type=None, bidir=False, path_model=None, device=None, INFO=True,
    use_batch_parallel=False,
    path_pretask_model=None, no_freeze=False,
):

    if model_type is None:
        if bidir:
            model_type = 'fedd-bidir'
        else:
            model_type = 'fedd'

    print(f"Initializing model_type: {model_type} from utils_models_v2v2_fedd.get_models_v2")
    
    
    # encoder_params_path = os.path.join(dir_config, 'encoder_config.yaml')
    # decoder_params_path = os.path.join(dir_config, 'decoder_config.yaml')
    
    # Check that the directory exists
    if not os.path.isdir(dir_config):
        raise FileNotFoundError(f"The configuration directory '{dir_config}' does not exist.")
    
    # Find the encoder config file: allowing any prefix before "encoder_config.yaml"
    encoder_files = glob.glob(os.path.join(dir_config, '*encoder_config.yaml'))
    if len(encoder_files) != 1:
        raise Exception(
            f"Expected exactly one encoder configuration file in '{dir_config}', "
            f"but found {len(encoder_files)}. Files found: {encoder_files}"
        )
    encoder_params_path = encoder_files[0]
    
    # Find the decoder config file: allowing any prefix before "decoder_config.yaml"
    decoder_files = glob.glob(os.path.join(dir_config, '*decoder_config.yaml'))
    if len(decoder_files) != 1:
        raise Exception(
            f"Expected exactly one decoder configuration file in '{dir_config}', "
            f"but found {len(decoder_files)}. Files found: {decoder_files}"
        )
    decoder_params_path = decoder_files[0]

    # Optionally, print or log the found paths for verification
    print("Encoder config path:", encoder_params_path)
    print("Decoder config path:", decoder_params_path)

    
    with open(encoder_params_path, "r") as f:
        encoder_params = yaml.safe_load(f)
    with open(decoder_params_path, "r") as f:
        decoder_params = yaml.safe_load(f)
    
    if encoder_params["img_size"] != img_size:
        print(f"Overwriting encoder_params['img_size'] (current value: {encoder_params['img_size']}) with img_size (from dataset): {img_size}")
        encoder_params["img_size"] = img_size
    if decoder_params["img_size"] != img_size:
        print(f"Overwriting decoder_params['img_size'] (current value: {decoder_params['img_size']}) with img_size (from dataset): {img_size}")
        decoder_params["img_size"] = img_size
    
    feature_encoder = FeatureEncoder(
        encoder_type=encoder_params["encoder_type"],
        encoder_params=encoder_params
    )

    ### deformation_decoder
    if not bidir:
        deformation_decoder = DeformationDecoder(
            decoder_type=decoder_params["decoder_type"],
            decoder_params=decoder_params
        )
    else:
        ### imports
        from deformation_inversion_layer.fixed_point_iteration import (
            AndersonSolver,
            AndersonSolverArguments,
            MaxElementWiseAbsStopCriterion,
            RelativeL2ErrorStopCriterion,
        )
        
        ### hard-coded configs
        ### consider moving this to the decoder_config?
        default_max_control_point_multiplier = 0.99
        forward_fixed_point_solver = AndersonSolver(
            stop_criterion=MaxElementWiseAbsStopCriterion(min_iterations=2, max_iterations=50, threshold=1e-2),
            arguments=AndersonSolverArguments(memory_length=4),
        )
        backward_fixed_point_solver = AndersonSolver(
            stop_criterion=RelativeL2ErrorStopCriterion(min_iterations=2, max_iterations=50, threshold=1e-2),
            arguments=AndersonSolverArguments(memory_length=4),
        )

        ### deformation_decoder
        from deformation_decoder.decoders.deformation_decoder_mysitreg import DeformationDecoderMySITReg
        deformation_decoder = DeformationDecoderMySITReg(
            decoder_params=decoder_params,
            forward_fixed_point_solver=forward_fixed_point_solver,
            backward_fixed_point_solver=backward_fixed_point_solver,
            max_control_point_multiplier=default_max_control_point_multiplier,
        )
    
    model = DIRNet(
        img_size,
        feature_encoder=feature_encoder,
        deformation_decoder=deformation_decoder,
        bidir=bidir,
        use_batch_parallel=use_batch_parallel
    )
    
    ###### load model weights
    if path_model is not None:
        # model.load_state_dict(torch.load(path_model)["state_dict"])
        model.load_state_dict(torch.load(path_model, weights_only=False)["state_dict"])
        if INFO:
            print(f"Loading fedd weights from: {path_model}")
    else:
        if INFO:
            print(f"path_model is None: model weights not loaded")
    
    ### load feature_encoder from pretrained pretask model
    if path_pretask_model is not None:
        # checkpoint = torch.load(path_pretask_model, map_location="cpu")
        checkpoint = torch.load(path_pretask_model, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        prefix = 'feature_encoder.'
        
        # Filter out only the keys that belong to the feature encoder.
        encoder_state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

        # Load the weights into the feature encoder.
        missing_keys, unexpected_keys = feature_encoder.load_state_dict(encoder_state_dict, strict=False)
        if missing_keys:
            print("Warning: missing keys:", missing_keys)
        if unexpected_keys:
            print("Warning: unexpected keys:", unexpected_keys)
            
        feature_encoder.freeze()
        if no_freeze:
            feature_encoder.unfreeze()
            
        if INFO:
            if no_freeze:
                print(f"Loading pretask_model's encoder from: {path_pretask_model} but unfreeze!!!")
            else:
                print(f"Loading pretask_model's encoder from: {path_pretask_model} and freeze!!!")
    
    ###### move to device
    if device is not None:
        model.to(device)
    
    ###### print number of trainable parameters
    if INFO:
        count_tot = count_trainable_parameters(model)
        count_enc = count_trainable_parameters(feature_encoder)
        count_dec = count_trainable_parameters(deformation_decoder)
        print(f"Number of trainable parameters model: {count_tot:,}")
        if count_enc != 0:
            print(f"\tFeature Encoder: {count_enc:,}; Deformation Decoder: {count_dec:,}; ratio dec/enc: {count_dec/count_enc:2f}")
        else:
            print(f"\tFeature Encoder: {count_enc:,}; Deformation Decoder: {count_dec:,} (encoder frozen)")
        
        # Check if total count matches the sum of the encoder and decoder
        if count_tot != (count_enc + count_dec):
            raise ValueError(
                f"Parameter count mismatch! model: {count_tot:,} != feature_encoder + deformation_decoder: {count_enc + count_dec:,}"
            )
        

    return model



        