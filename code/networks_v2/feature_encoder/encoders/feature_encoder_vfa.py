# feature_encoder/encoders/feature_encoder_vfa.py

import torch
import torch.nn as nn

from ..registries import register_encoder, BLOCK_REGISTRY

@register_encoder('cnn_vfa')
class FeatureEncoderVFA(nn.Module):
    def __init__(self, **encoder_params):
        super().__init__()

        self.img_size = encoder_params['img_size']
        self.freeze = encoder_params['freeze']
        self.project = encoder_params['project']
        self.path_checkpoint = encoder_params['path_checkpoint']

        print('self.freeze: ', self.freeze)
        print('self.project: ', self.project)
        print('self.path_checkpoint: ', self.path_checkpoint)
        

        model_type = 'bs3_vfa_v0'
        # path_checkpoint = '/database/wavereg/oasis_v1_run1/250219_gpu7_rs42_bs3_vfa_v0_run1_20250220001148/checkpoint_0100/dice0.8095_epoch0098.pth.tar'
        import os
        import sys
        my_src_dir = os.path.abspath(
            '/homebase/DL_projects/wavereg/code'
        )
        sys.path.insert(0, my_src_dir)
        import utils_models_v2_baseline as utils_models_v2_baseline
        self.model = utils_models_v2_baseline.get_models_v2(self.img_size, model_type, self.path_checkpoint)

        if self.freeze:
            print('Freezing model parameters in FeatureEncoderVFA')
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print('Model parameters in FeatureEncoderVFA is not frozen')
        if self.project:
            print('Projection head is used in FeatureEncoderVFA')
        else:
            print('Projection head is DISABLED in FeatureEncoderVFA')

        self.feature_encoder = self.model.unet_model.encoder
        self.projectors = self.model.unet_model.decoder.project
            
    def forward(self, x):

        list_features = self.feature_encoder(x)
        
        if self.project:
            list_projs = []
            for idx in range(len(list_features)):
                proj = self.projectors[idx](list_features[idx])
                list_projs.append(proj)
            return list_projs
        else:
            return list_features
        
            