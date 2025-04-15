
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.init as init

from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Any, Optional

import sys
sys.path.insert(0, '/homebase/DL_projects/wavereg/code')
import utils_warp # utils_warp.SpatialTransformer


class DIRNet(nn.Module):
    """
    Registration network wrapper
    
    Notes:
        1). SpatialTransformer:
            updated SpatialTransformer: utils_warp.SpatialTransformer
        2). DVF in this context means deformation vector field.
            In other contexts (e.g. diffeomorphic), DVF is used for dense velocity field and DDF for dense displacement field
            DVF in our context is the same as dense displacement field.
        3). Coding notes:
            use x/y or x1/x2?
            currently use x/y notation
    """
    def __init__(
        self,
        inshape: Tuple[int, int, int],
         # Option 1: a unified registration model.
        unet_model: Optional[nn.Module] = None,
        init_method_unet: str = None,
        # Option 2: separate feature encoder and deformation decoder.
        feature_encoder: Optional[nn.Module] = None,
        deformation_decoder: Optional[nn.Module] = None,
        use_batch_parallel: Optional[bool] = True,
        bidir: bool = False,
        persistent_grid: bool = False,
        INFO: bool = True,
        DEBUG: bool = False
        ) -> None:

        super().__init__()

        self.INFO = INFO
        self.DEBUG = DEBUG

        self.inshape = inshape
        self.bidir = bidir

        ### set up SpatialTransformer
        self.transformer = utils_warp.SpatialTransformer(self.inshape, persistent=persistent_grid)

        ### Mode selection: check that either unet_model is provided or both feature_encoder and deformation_decoder.
        if unet_model is None and (feature_encoder is None or deformation_decoder is None):
            raise ValueError("Either unet_model or both feature_encoder and deformation_decoder must be provided.")

        self.unet_model = unet_model
        self.feature_encoder = feature_encoder
        self.deformation_decoder = deformation_decoder
        self.use_batch_parallel = use_batch_parallel
        print(f"INFO: in DIRNet unet_model's use_batch_parallel is set to: {self.use_batch_parallel}")
        
        ### weight initialization
        if self.unet_model is not None:
            if init_method_unet == 'FORCE_OFF':
                if self.INFO:
                    print("INFO: in DIRNet unet_model's initialize_weights() method FORCE_OFF.")
            else:
                if hasattr(self.unet_model, 'initialize_weights'):
                    try:
                        self.unet_model.initialize_weights(init_method_unet)
                        if self.INFO:
                            print("INFO: unet_model.initialize_weights(init_method_unet) has been called.")
                    except TypeError:
                        self.unet_model.initialize_weights()
                        if self.INFO:
                            print("INFO: unet_model.initialize_weights() has been called without init_method_unet.")
                else:
                    if self.INFO:
                        print("INFO: unet_model does not have an initialize_weights() method.")
        else:
            if self.INFO:
                print("INFO: Using separate feature_encoder and deformation_decoder. Each model handles its own initialization.")

                
    def forward(
        self, 
        moving: torch.FloatTensor, 
        fixed: torch.FloatTensor, 
        unet_kwargs: Optional[Dict[str, Any]] = None, # UNet
        return_intermediates: bool = False, # FE+DD
    ) -> Union[
        Tuple[torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor, Any],
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, Any]
        ]:
        """
        Inputs:
            moving: moving image tensor (moving/source) -> x
            fixed: fixed image tensor (fixed/target) -> y
            unet_kwargs: optional dictionary of additional arguments for unet_model
        Outputs (if self.bidir is False):
            deformed_x: deformed moving image tensor (deformed/moved/warped) -> y_pred
            dvf_forward: deformation vector field (dvf/warp)
            extra: any additional data returned by unet_model (optional)
            
        Outputs (if self.bidir is True):
            deformed_x: deformed moving image tensor -> y_pred
            dvf_forward: deformation vector field
            deformed_y: deformed fixed image tensor -> x_pred
            dvf_reverse: reverse deformation vector field
            extra: any additional data returned by unet_model (optional)
        """
        x = moving
        y = fixed
        
        

        ### unified unet
        if self.unet_model is not None:
            # concatenate along the channel (NCHWD) dimension
            input_unet = torch.cat([x, y], dim=1)
            
            if unet_kwargs is None:
                unet_kwargs = {}  # Ensure unet_kwargs is a dictionary
            
            # Register (forward pass to predict dvf and possibly other outputs)
            reg_outputs = self.unet_model(input_unet, **unet_kwargs)
        ### feature_encoder + deformation_decoder
        else:
            if self.use_batch_parallel: # use batch dimension to calculate in parallel
                # Assume x and y are your inputs with shape (batch_size, 1, H, W, D)
                # For your current example, batch_size might be 1 for each.
                combined_input = torch.cat([x, y], dim=0)
                
                # Forward pass through the feature encoder
                combined_features = self.feature_encoder(combined_input)
                # combined_features is now a list of tensors, each with shape (batch_x + batch_y, ...)

                reg_outputs = self.deformation_decoder(combined_features, return_intermediates=return_intermediates) # moving, fixed
                
                # # Determine the split point (here x and y might each have batch size of 1)
                # batch_x = x.size(0)
                
                # # Split each tensor in the list back into features for x and y
                # list_features_x = [feat[:batch_x] for feat in combined_features]
                # list_features_y = [feat[batch_x:] for feat in combined_features]
                
                # reg_outputs = self.deformation_decoder(list_features_x, list_features_y, return_intermediates=return_intermediates) # moving, fixed
            else: # calculate sequentially
                list_features_x = self.feature_encoder(x)
                list_features_y = self.feature_encoder(y)
                reg_outputs = self.deformation_decoder(list_features_x, list_features_y, return_intermediates=return_intermediates) # moving, fixed
            
        # Unidirectional registration
        if not self.bidir:
            # Handle extra outputs
            if isinstance(reg_outputs, (tuple, list)) and len(reg_outputs) > 1:
                dvf_forward = reg_outputs[0]
                extra = reg_outputs[1:]
            else:
                dvf_forward = reg_outputs[0] if isinstance(reg_outputs, (tuple, list)) else reg_outputs
                extra = None
            
            # Warp the moving image (x) with dvf
            # default settings: (mode='bilinear', padding_mode='border', align_corners=True)
            deformed_x = self.transformer(x, dvf_forward)
            
            # Return deformed image, dvf, and any additional outputs
            if extra is None or len(extra) == 0:
                return deformed_x, dvf_forward  # Tuple length = 2
            else:
                return deformed_x, dvf_forward, extra  # Tuple length = 3

        # Bidirectional registration (inverse consistency)
        else:
            # Handle extra outputs
            if not isinstance(reg_outputs, (list, tuple)) or len(reg_outputs) < 2:
                raise ValueError("For bidirectional registration, unet_model must output at least 2 values (dvf_forward and dvf_reverse).")
            elif len(reg_outputs) > 2:
                dvf_forward, dvf_reverse = reg_outputs[0:2]
                extra = reg_outputs[2:]  # Any additional outputs
            else:  # exactly 2 outputs
                dvf_forward, dvf_reverse = reg_outputs
                extra = None

            # Warp the images
            deformed_x = self.transformer(x, dvf_forward)
            deformed_y = self.transformer(y, dvf_reverse)

            # Return deformed images, dvfs, and any additional outputs
            if extra is None or len(extra) == 0:
                return deformed_x, dvf_forward, deformed_y, dvf_reverse  # Tuple length = 4
            else:
                return deformed_x, dvf_forward, deformed_y, dvf_reverse, extra  # Tuple length = 5
           
