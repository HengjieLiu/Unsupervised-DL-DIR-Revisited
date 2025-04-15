

import math
from math import exp

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import Variable

"""
Current situation as of 25/03/31
    NCC_vfa vs NCC_vfa_fast
        no apparent speed improvement when tested on sitreg
        the val dice has a bit decrease for NCC_vfa_fast
    NCC_vxm_fast still has bugs ..


List of functions:
Note: 
    1) All ncc losses are added 1 to shift to 0-1 in this script!!!
    2) torch.nn.functional is imported as nnf instead of F
    3) use nn.Module （need super().__init__()) and forward

gaussian (for NCC_gauss)
NCC_vxm
    https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/losses.py
NCC_vxm_fast
    https://github.com/xi-jia/FastLNCC/blob/main/FastLNCC.py
NCC_gauss:
    https://github.com/junyuchen245/TransMorph_DCA/blob/09e32758e5788a358c8543ca61fe0cc76e75b0dc/OASIS/losses.py
NCC_vfa
    https://github.com/yihao6/vfa/blob/main/vfa/losses/ncc_loss.py
NCC_vfa_fast

"""




def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


class NCC_vxm(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super().__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        # conv_fn = getattr(F, 'conv%dd' % ndims)
        conv_fn = getattr(nnf, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return 1-torch.mean(cc)


class NCC_vxm_fast(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super().__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        # sum_filt = torch.ones([1, 1, *win]).to("cuda")
        sum_filt = torch.ones([5, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        # conv_fn = getattr(F, 'conv%dd' % ndims)
        conv_fn = getattr(nnf, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji
        
        all_five = torch.cat((Ii, Ji, I2, J2, IJ),dim=1)
        all_five_conv = conv_fn(all_five, sum_filt, stride=stride, padding=padding, groups=5)
        I_sum, J_sum, I2_sum, J2_sum, IJ_sum = torch.split(all_five_conv, 1, dim=1)
        
        # I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        # J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        # I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        # J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        # IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        # compute cross correlation
        # win_size = np.prod(win)
        # u_I = I_sum / win_size
        # u_J = J_sum / win_size

        # cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        # I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        # J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size


        # compute cross correlation
        win_size = np.prod(self.win)

        cross = IJ_sum - J_sum/win_size*I_sum
        I_var = I2_sum - I_sum/win_size*I_sum
        J_var = J2_sum - J_sum/win_size*J_sum

        
        cc = cross * cross / (I_var * J_var + 1e-5)

        return 1-torch.mean(cc)
    

class NCC_gauss(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss via Gaussian kernel
    """

    def __init__(self, win=9):
        super(NCC_gauss, self).__init__()
        self.win = [win]*3
        self.filt = self.create_window_3D(win, 1).to("cuda")

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window_3D(self, window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                      window_size).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
        return window

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # compute filters
        pad_no = math.floor(self.win[0] / 2)

        # get convolution function
        # conv_fn = getattr(F, 'conv%dd' % ndims)
        conv_fn = getattr(nnf, 'conv%dd' % ndims)

        # compute CC squares
        mu1 = conv_fn(Ii, self.filt, padding=pad_no)
        mu2 = conv_fn(Ji, self.filt, padding=pad_no)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv_fn(Ii * Ii, self.filt, padding=pad_no) - mu1_sq
        sigma2_sq = conv_fn(Ji * Ji, self.filt, padding=pad_no) - mu2_sq
        sigma12 = conv_fn(Ii * Ji, self.filt, padding=pad_no) - mu1_mu2

        cc = (sigma12 * sigma12 + 1e-5)/(sigma1_sq * sigma2_sq + 1e-5)
        return 1-torch.mean(cc)


# class SingleScaleNCC(nn.Module):
    # def __init__(self, window_size, **kwargs):
class NCC_vfa(nn.Module):
    def __init__(self, window_size=9, **kwargs):
        super().__init__()
        if isinstance(window_size, int):
            self.window_size = [window_size]
        else:
            self.window_size = window_size

    def forward(self, pred, target):
        """ LNCC loss
            modified based on https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/TransMorph/losses.py
        """
        Ii = target
        Ji = pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = self.window_size * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(pred.device) / np.prod(win)

        pad_no = win[0] // 2

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(nnf, 'conv%dd' % ndims)

        # compute CC squares
        mu1 = conv_fn(Ii, sum_filt, padding=padding, stride=stride)
        mu2 = conv_fn(Ji, sum_filt, padding=padding, stride=stride)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv_fn(Ii * Ii, sum_filt, padding=padding, stride=stride) - mu1_sq
        sigma2_sq = conv_fn(Ji * Ji, sum_filt, padding=padding, stride=stride) - mu2_sq
        sigma12 = conv_fn(Ii * Ji, sum_filt, padding=padding, stride=stride) - mu1_mu2

        eps = torch.finfo(sigma12.dtype).eps
        cc = (sigma12 * sigma12) / torch.clamp(sigma1_sq * sigma2_sq, min=eps)
        # return - torch.mean(cc)
        return 1 - torch.mean(cc)



class NCC_vfa_fast(nn.Module):
    def __init__(self, window_size=9, **kwargs):
        super().__init__()
        # if an integer is provided, convert it to a list
        if isinstance(window_size, int):
            self.window_size = [window_size]
        else:
            self.window_size = window_size

    def forward(self, pred, target):
        # In this formulation, target is Ii and pred is Ji.
        Ii = target
        Ji = pred

        # Determine the number of spatial dimensions (expects [B, 1, ...])
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], f"Volumes should be 1 to 3 dimensions. Found: {ndims}"

        # Create the window dimensions; e.g., for 2D with window_size=9, win = [9, 9]
        win = self.window_size * ndims
        win_size = np.prod(win)

        # Create a normalized filter (ones divided by the number of elements) for each of the 5 convolutions.
        # Here we have 5 groups (for mu1, mu2, conv(I^2), conv(J^2), conv(IJ))
        filt = torch.ones([5, 1, *win], device=pred.device) / win_size

        # Set padding so that the convolution is “centered”
        pad_no = win[0] // 2
        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:  # 3D
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # Get the appropriate convolution function (e.g., conv1d, conv2d, or conv3d)
        conv_fn = getattr(nnf, 'conv%dd' % ndims)

        # Compute the squared and cross terms
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # Concatenate the inputs along the channel dimension.
        # This gives a tensor of shape [B, 5, ...] where each channel corresponds to one of the terms.
        all_five = torch.cat((Ii, Ji, I2, J2, IJ), dim=1)

        # Perform one grouped convolution (groups=5) so that each channel is convolved with its corresponding filter.
        all_five_conv = conv_fn(all_five, filt, padding=padding, stride=stride, groups=5)

        # Split the result into the five local averages
        mu1, mu2, conv_I2, conv_J2, conv_IJ = torch.split(all_five_conv, 1, dim=1)

        # Compute the local variances and covariance:
        # variance = E[I^2] - (E[I])^2 and covariance = E[I*J] - E[I]E[J]
        sigma1_sq = conv_I2 - mu1 * mu1
        sigma2_sq = conv_J2 - mu2 * mu2
        sigma12   = conv_IJ - mu1 * mu2

        # Use a small epsilon for numerical stability
        eps = torch.finfo(sigma12.dtype).eps
        cc = (sigma12 * sigma12) / (sigma1_sq * sigma2_sq + eps)

        # Return the loss (1 - mean local NCC)
        return 1 - torch.mean(cc)