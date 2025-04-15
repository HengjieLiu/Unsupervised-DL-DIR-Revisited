

"""
List of classes/functions:
    count_trainable_parameters
    report_gpu_memory
"""

######### general imports
import torch
import torch.nn as nn
import torch.nn.functional as nnf


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def report_gpu_memory(device='cuda'):
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved  = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved  = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    print(f"GPU Memory (GB): Allocated: {allocated:.4f}, Reserved: {reserved:.4f}, "
          f"Peak Allocated: {peak_allocated:.4f}, Peak Reserved: {peak_reserved:.4f}")
