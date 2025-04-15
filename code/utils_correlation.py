"""
Correlation calculation

List of classes:
    GlobalCorrTorch
    WinCorrTorch
    WinCorrTorch_VFA
    WinCorrTorch_Conv (try to speed up using conv, but not successful)
Both WinCorrTorch and GlobalCorrTorch are based on:
    https://github.com/BailiangJ/rethink-reg/blob/main/models/networks/correlation.py
        WinCorrTorch3D
        GlobalCorrTorch3D
    But they are modified to fit both 2D and 3D ...

##################################
Test script in:
    homebase/DL_projects/wavereg/code/unittest_corr/test_corr_0403_4_compare_all.ipynb
        WinCorrTorch3D_VFA is the fastest one
        need to set padding_mode to constant to align with others ...

##################################
NOTE:
    currently if I call any WinCorr***(x, y), WinCorrTorch and WinCorrTorch_VFA give the same results
    But there is an order mismatch:
        in Rethink-Reg:
            corr = self.corr[i](src_feats[i], tgt_feats[i]) # src/moving, tgt/fixed
            the order is moving, fixed
            and in WinCorrTorch3D, the 2nd input is shifting around
        in VFA:
            MOVING IS SHIFTED AROUND 3x3x3
            # Extract candidate patches from the padded moving feature map.
            K = self.get_candidate_from_tensor(moving_feat_padded)  # shape: [B, spatial..., patch, C]
            # Rearrange fixed feature map to align dimensions for matrix multiplication.
            # For 3D, fixed_feat: [B, C, D, H, W] -> [B, D, H, W, C] and add a singleton patch dim.
            Q = fixed_feat.permute(*self.permute_order).unsqueeze(-2)  # [B, spatial..., 1, C]
        
        
    
Modifications:
    ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf



class GlobalCorrTorch(nn.Module):
    def __init__(self, dim: int = 3):
        """
        Args:
            dim (int): Number of spatial dimensions (2 or 3).
        """
        super().__init__()
        if dim not in [2, 3]:
            raise ValueError("dim must be 2 or 3.")
        self.dim = dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape
        if self.dim == 3:
            # Expecting [B, C, D, H, W]
            b, c, d, h, w = x.shape
            x_flat = x.view(b, c, -1)  # flatten D, H, W
            y_flat = y.view(b, c, -1)
            corr = torch.einsum('bci, bcj -> bji', x_flat, y_flat)
            corr *= (c ** -0.5)
            return corr.view(b, -1, d, h, w)
        else:
            # 2D: Expecting [B, C, H, W]
            b, c, h, w = x.shape
            x_flat = x.view(b, c, -1)  # flatten H, W
            y_flat = y.view(b, c, -1)
            corr = torch.einsum('bci, bcj -> bji', x_flat, y_flat)
            corr *= (c ** -0.5)
            return corr.view(b, -1, h, w)


class WinCorrTorch(nn.Module):
    def __init__(self, dim: int = 3, radius: int = 1):
        """
        Args:
            dim (int): Number of spatial dimensions (2 or 3).
            radius (int): Radius for the local window.
        """
        super().__init__()
        if dim not in [2, 3]:
            raise ValueError("dim must be 2 or 3.")
        self.dim = dim
        self.radius = radius
        self.win_size = 2 * radius + 1

        # Choose the appropriate constant padding.
        if self.dim == 3:
            self.padding = nn.ConstantPad3d(radius, 0)
        else:
            self.padding = nn.ConstantPad2d(radius, 0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape
        if self.dim == 3:
            # 3D: [B, C, D, H, W]
            b, c, d, h, w = x.shape
            y_padded = self.padding(y)
            # Create a meshgrid for 3 dimensions.
            offsets = torch.meshgrid(
                [torch.arange(0, self.win_size) for _ in range(3)], indexing='ij'
            )
            # Compute correlation for each offset in 3D.
            corr = torch.cat(
                [
                    torch.sum(
                        x * y_padded[:, :, dz:dz + d, dy:dy + h, dx:dx + w],
                        dim=1, keepdim=True
                    )
                    for dz, dy, dx in zip(
                        offsets[0].flatten(), offsets[1].flatten(), offsets[2].flatten()
                    )
                ],
                dim=1
            )
            corr *= (c ** -0.5)
            return corr
        else:
            # 2D: [B, C, H, W]
            b, c, h, w = x.shape
            y_padded = self.padding(y)
            # Create a meshgrid for 2 dimensions.
            offsets = torch.meshgrid(
                [torch.arange(0, self.win_size) for _ in range(2)], indexing='ij'
            )
            # Compute correlation for each offset in 2D.
            corr = torch.cat(
                [
                    torch.sum(
                        x * y_padded[:, :, dy:dy + h, dx:dx + w],
                        dim=1, keepdim=True
                    )
                    for dy, dx in zip(
                        offsets[0].flatten(), offsets[1].flatten()
                    )
                ],
                dim=1
            )
            corr *= (c ** -0.5)
            return corr




class WinCorrTorch_VFA(nn.Module):
    def __init__(self, dim: int = 3, radius: int = 1, stride: int = 1, padding_mode: str = 'replicate', normalize: bool = True):
        """
        Args:
            kernel (int): Kernel size to extract patches (default: 3).
            stride (int): Stride for patch extraction (default: 1).
            dim (int): Number of spatial dimensions (2 or 3). 
                       For 3D, input shape is [B, C, D, H, W]; for 2D, [B, C, H, W].
            padding_mode (str): Padding mode passed to F.pad (default: 'replicate').
            normalize (bool): Whether to normalize the correlation by scaling with (C ** -0.5).
        """
        super().__init__()
        
        self.dim = dim
        self.kernel = 2 * radius + 1
        self.stride = stride
        self.padding_mode = padding_mode
        self.normalize = normalize
        
        # Automatically compute the padding for "same" output size:
        self.padding = (self.kernel - 1) // 2

        # Set the permutation order to rearrange fixed feature maps.
        # For 3D: from [B, C, D, H, W] to [B, D, H, W, C].
        # For 2D: from [B, C, H, W] to [B, H, W, C].
        if self.dim == 3:
            self.permute_order = (0, 2, 3, 4, 1)
        elif self.dim == 2:
            self.permute_order = (0, 2, 3, 1)
        else:
            raise ValueError("dim must be 2 or 3.")

    def get_candidate_from_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches from tensor x.

        For 3D:
            x: [B, C, D, H, W] -> returns [B, D, H, W, (kernel**3), C].
        For 2D:
            x: [B, C, H, W] -> returns [B, H, W, (kernel**2), C].
        """
        if self.dim == 3:
            # Unfold the last three dims: D, H, W.
            patches = (x.unfold(2, self.kernel, self.stride)
                         .unfold(3, self.kernel, self.stride)
                         .unfold(4, self.kernel, self.stride))
            # Merge the kernel dimensions into one.
            patches = patches.flatten(start_dim=5)
            # Rearrange so that the patch dimension is last and channels are last.
            token = patches.permute(0, 2, 3, 4, 5, 1)  # [B, D, H, W, patch, C]
        elif self.dim == 2:
            patches = x.unfold(2, self.kernel, self.stride).unfold(3, self.kernel, self.stride)
            patches = patches.flatten(start_dim=4)
            token = patches.permute(0, 2, 3, 4, 1)  # [B, H, W, patch, C]
        return token

    def forward(self, fixed_feat: torch.Tensor, moving_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fixed_feat: Feature map from the fixed image, shape [B, C, ...].
            moving_feat: Feature map from the moving image, shape [B, C, ...].

        Returns:
            Correlation volume. For 3D, output shape is [B, (kernel**3), D, H, W];
            for 2D, output shape is [B, (kernel**2), H, W].
        """
        # Compute the symmetric padding tuple.
        # For dim==3, pad order is (w_left, w_right, h_top, h_bottom, d_front, d_back).
        pad = [self.padding] * (self.dim * 2)
        moving_feat_padded = nnf.pad(moving_feat, pad=tuple(pad), mode=self.padding_mode)

        # Extract candidate patches from the padded moving feature map.
        K = self.get_candidate_from_tensor(moving_feat_padded)  # shape: [B, spatial..., patch, C]
        # Rearrange fixed feature map to align dimensions for matrix multiplication.
        # For 3D, fixed_feat: [B, C, D, H, W] -> [B, D, H, W, C] and add a singleton patch dim.
        Q = fixed_feat.permute(*self.permute_order).unsqueeze(-2)  # [B, spatial..., 1, C]

        # Compute correlation as the dot product between Q and candidate patches.
        # Here, K.transpose(-1, -2) changes shape to [B, spatial..., C, patch].
        attention = torch.matmul(Q, K.transpose(-1, -2))  # [B, spatial..., 1, patch]

        # Normalize by channel count if requested.
        if self.normalize:
            C = fixed_feat.size(1)
            attention = attention * (C ** -0.5)

        # Remove the singleton patch dimension.
        out = attention.squeeze(-2)  # [B, spatial..., patch]

        # Optionally, rearrange dimensions so that the patch dimension comes right after the batch.
        if self.dim == 3:
            # From [B, D, H, W, patch] to [B, patch, D, H, W].
            out = out.permute(0, 4, 1, 2, 3)
        elif self.dim == 2:
            # From [B, H, W, patch] to [B, patch, H, W].
            out = out.permute(0, 3, 1, 2)
        return out


class WinCorrTorch_Conv(nn.Module):
    def __init__(self, dim: int = 3, radius: int = 1, normalize: bool = True):
        """
        Args:
            dim (int): Number of spatial dimensions (2 or 3).
            radius (int): Radius of the local window.
            normalize (bool): If True, scales the correlation by (C ** -0.5).
        """
        super().__init__()
        if dim not in [2, 3]:
            raise ValueError("dim must be 2 or 3.")
        self.dim = dim
        self.radius = radius
        self.kernel_size = 2 * radius + 1
        # For 3D use kernel_volume = kernel_size^3; for 2D use kernel_size^2.
        self.kernel_volume = self.kernel_size ** (3 if self.dim == 3 else 2)
        self.normalize = normalize

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fixed: Fixed image features, shape [B, C, ...].
            moving: Moving image features, shape [B, C, ...].
        Returns:
            Correlation volume with shape [B, kernel_volume, ...] where ... are spatial dimensions.
        """
        B, C = fixed.shape[:2]
        if self.dim == 3:
            # Expected shapes: fixed and moving: [B, C, D, H, W]
            _, _, D, H, W = fixed.shape
            weight = torch.zeros(
                (C * self.kernel_volume, 1, self.kernel_size, self.kernel_size, self.kernel_size),
                device=moving.device,
                dtype=moving.dtype,
            )
            offsets = [(i, j, k) for i in range(self.kernel_size)
                                for j in range(self.kernel_size)
                                for k in range(self.kernel_size)]
            for c in range(C):
                for idx, (i, j, k) in enumerate(offsets):
                    filter_index = c * self.kernel_volume + idx
                    weight[filter_index, 0, i, j, k] = 1.0

            patches = nnf.conv3d(
                moving, weight, bias=None, stride=1, padding=self.radius, groups=C
            )
            # Reshape: [B, C * kernel_volume, D, H, W] -> [B, kernel_volume, C, D, H, W]
            patches = patches.view(B, C, self.kernel_volume, D, H, W).permute(0, 2, 1, 3, 4, 5)
            fixed_expanded = fixed.unsqueeze(1)  # [B, 1, C, D, H, W]
            correlation = (fixed_expanded * patches).sum(dim=2)  # [B, kernel_volume, D, H, W]
        else:
            # 2D: Expected shapes: fixed and moving: [B, C, H, W]
            _, _, H, W = fixed.shape
            weight = torch.zeros(
                (C * self.kernel_volume, 1, self.kernel_size, self.kernel_size),
                device=moving.device,
                dtype=moving.dtype,
            )
            offsets = [(i, j) for i in range(self.kernel_size)
                             for j in range(self.kernel_size)]
            for c in range(C):
                for idx, (i, j) in enumerate(offsets):
                    filter_index = c * self.kernel_volume + idx
                    weight[filter_index, 0, i, j] = 1.0

            patches = nnf.conv2d(
                moving, weight, bias=None, stride=1, padding=self.radius, groups=C
            )
            # Reshape: [B, C * kernel_volume, H, W] -> [B, kernel_volume, C, H, W]
            patches = patches.view(B, C, self.kernel_volume, H, W).permute(0, 2, 1, 3, 4)
            fixed_expanded = fixed.unsqueeze(1)  # [B, 1, C, H, W]
            correlation = (fixed_expanded * patches).sum(dim=2)  # [B, kernel_volume, H, W]

        if self.normalize:
            correlation *= (C ** -0.5)
        return correlation