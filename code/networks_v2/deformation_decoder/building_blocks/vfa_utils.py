
import torch
import torch.nn as nn
import torch.nn.functional as nnf

"""

List of functions/classes:
    def identity_grid_like
        https://github.com/yihao6/vfa/blob/main/vfa/utils/utils.py
    class Attention
        https://github.com/yihao6/vfa/blob/main/vfa/models/vfa.py
"""


def identity_grid_like(tensor, normalize, padding=0):
    """
    Direct copy from:
        https://github.com/yihao6/vfa/blob/main/vfa/utils/utils.py
    """
    '''return the identity grid for the input 2D or 3D tensor'''
    with torch.inference_mode():
        dims = tensor.shape[2:]
        if isinstance(padding, int):
            pads = [padding for j in range(len(dims))]
        else:
            pads = padding
        vectors = [torch.arange(start=0-pad, end=dim+pad) for (dim,pad) in zip(dims, pads)]

        try:
            grids = torch.meshgrid(vectors, indexing='ij')
        except TypeError:
            # compatible with old pytorch version
            grids = torch.meshgrid(vectors)
        grid = torch.stack(grids).unsqueeze(0).type(torch.float32)

        if normalize:
            in_dims = torch.tensor(dims).view(-1, len(dims), *[1 for x in range(len(dims))])
            grid = grid / (in_dims - 1) * 2 - 1

        grid = grid.to(tensor.device).repeat(tensor.shape[0],1,*[1 for j in range(len(dims))])
    return grid.clone()


class Attention(nn.Module):
    """
    Direct copy from:
        https://github.com/yihao6/vfa/blob/main/vfa/models/vfa.py
    """
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, temperature):
        '''Expect input dimensions: [batch, *, feature]'''
        if temperature is None:
            temperature = key.size(-1) ** 0.5
        attention = torch.matmul(query, key.transpose(-1, -2)) / temperature
        attention = self.softmax(attention)
        x = torch.matmul(attention, value)
        return x

# class Attention(nn.Module):
#     """
#     # An Attention module similar to that used in VFA (very simple version) 
#     #  with GPT commentsd
#     """
#     def __init__(self):
#         super().__init__()
#         self.softmax = nn.Softmax(dim=-1)
        
#     def forward(self, query, key, value, temperature):
#         # query: [B, N, 1, C]
#         # key:   [B, N, P, C]   (here P is the patch dimension, in our simple case P==1)
#         # value: [N, C]         (pre-computed radial vector field for displacement)
#         if temperature is None:
#             temperature = key.size(-1) ** 0.5
#         # Compute similarity: [B, N, 1, P]
#         attn_scores = torch.matmul(query, key.transpose(-1, -2)) / temperature
#         attn = self.softmax(attn_scores)
#         # Multiply attention weights with the value.
#         # First, expand the value to [1, N, P, C] then multiply and sum over the patch dimension (P)
#         value_exp = value.unsqueeze(0)  # shape: [1, N, C]
#         # For convenience, add a singleton patch dimension:
#         value_exp = value_exp.unsqueeze(2)  # [1, N, 1, C]
#         # Now weighted sum over the patch dimension:
#         local_disp = torch.matmul(attn, value_exp)  # shape: [B, N, 1, C]
#         return local_disp