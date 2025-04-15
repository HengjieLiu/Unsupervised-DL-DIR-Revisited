

import math
from math import exp

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import Variable


class Dice_vxm(nn.Module):
    """
    N-D dice for segmentation
    
    Modification:
        change into class and add init and use forward
        add 1
        switch order y_pred, y_true (should not matter)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return 1 - dice

# class Dice:
#     """
#     N-D dice for segmentation
#     """
#
#     def loss(self, y_true, y_pred):
#         ndims = len(list(y_pred.size())) - 2
#         vol_axes = list(range(2, ndims + 2))
#         top = 2 * (y_true * y_pred).sum(dim=vol_axes)
#         bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
#         dice = torch.mean(top / bottom)
#         return -dice
    