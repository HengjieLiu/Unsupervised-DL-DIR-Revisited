"""
List of functions and classes:
    def save_checkpoint
    
"""

import os
import glob
import re
import torch
import time
import random
from natsort import natsorted

def save_checkpoint(state, 
                    save_dir='models', 
                    filename='checkpoint.pth.tar', 
                    max_model_num=10,
                    metric_key='best_val_dice',
                    metric_regex_prefix='dice',
                    maximize=True,
                    DEBUG=False):
    """
    NOTE：
        Currently it still only work for positive loss, so make sure the loss is positive
        Maybe it works for negative, but I just haven't tested
        
    
    Save a checkpoint only if the current model is among the best according to a given metric.
    
    Parameters:
      state (dict): Dictionary containing the model state and metrics.
      save_dir (str): Directory where checkpoints are saved.
      filename (str): Filename for the new checkpoint.
      max_model_num (int): Maximum number of checkpoints to keep for this metric.
      metric_key (str): Key in `state` holding the metric value 
                        (e.g., 'best_val_dice', 'best_val_loss_tot', or 'best_val_loss_sim').
      metric_regex_prefix (str): The literal prefix used in the filename for this metric (e.g., 'dice', 'val_loss_tot').
      maximize (bool): If True, higher metric values are better (e.g., Dice). 
                       If False, lower metric values are better (e.g., loss).
      DEBUG (bool): If True, print DEBUG messages showing what the function is doing.
    """
    # Define a universal float regex that matches both fixed-point and scientific notation.
    FLOAT_REGEX = r'([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)'
    # Build the complete regex pattern using the prefix.
    metric_regex = metric_regex_prefix + FLOAT_REGEX
    
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, filename)
    
    # Retrieve the current metric; set a default if not found.
    current_metric = state.get(metric_key, -float('inf') if maximize else float('inf'))
    if DEBUG:
        print(f"DEBUG: Current {metric_key} = {current_metric:.4f} for checkpoint {filename}")
    
    # List only checkpoints that match the specified metric using the regex.
    all_files = glob.glob(os.path.join(save_dir, '*'))
    model_lists = natsorted([
        f for f in all_files
        if re.search(metric_regex, os.path.basename(f))
    ])
    
    # If we have fewer than max_model_num checkpoints, simply save the new checkpoint.
    if len(model_lists) < max_model_num:
        if DEBUG:
            print(f"DEBUG: Fewer than {max_model_num} models exist for metric {metric_key}. Saving checkpoint to {model_path}.")
        torch.save(state, model_path)
    else:
        # Helper to extract the metric value from a checkpoint filename.
        def extract_metric(fname):
            match = re.search(metric_regex, os.path.basename(fname))
            if match:
                return float(match.group(1))
            else:
                return -float('inf') if maximize else float('inf')
        
        if maximize:
            # For metrics where higher is better, identify the checkpoint with the smallest metric.
            worst_checkpoint = min(model_lists, key=extract_metric)
            worst_metric = extract_metric(worst_checkpoint)
            if DEBUG:
                print(f"DEBUG: Found {len(model_lists)} checkpoints for metric {metric_key}.")
                print(f"DEBUG: Worst checkpoint is {os.path.basename(worst_checkpoint)} with metric value {worst_metric:.4f}.")
            if current_metric > worst_metric:
                if DEBUG:
                    print(f"DEBUG: Current metric {current_metric:.4f} is higher than worst {worst_metric:.4f}.")
                    print(f"DEBUG: Removing {os.path.basename(worst_checkpoint)} and saving new checkpoint to {model_path}.")
                os.remove(worst_checkpoint)
                torch.save(state, model_path)
            else:
                if DEBUG:
                    print(f"DEBUG: Current metric {current_metric:.4f} is not higher than worst {worst_metric:.4f}. Skipping save.")
        else:
            # For metrics where lower is better, identify the checkpoint with the highest metric.
            worst_checkpoint = max(model_lists, key=extract_metric)
            worst_metric = extract_metric(worst_checkpoint)
            if DEBUG:
                print(f"DEBUG: Found {len(model_lists)} checkpoints for metric {metric_key}.")
                print(f"DEBUG: Worst checkpoint is {os.path.basename(worst_checkpoint)} with metric value {worst_metric:.4f}.")
            if current_metric < worst_metric:
                if DEBUG:
                    print(f"DEBUG: Current metric {current_metric:.4f} is lower than worst {worst_metric:.4f}.")
                    print(f"DEBUG: Removing {os.path.basename(worst_checkpoint)} and saving new checkpoint to {model_path}.")
                os.remove(worst_checkpoint)
                torch.save(state, model_path)
            else:
                if DEBUG:
                    print(f"DEBUG: Current metric {current_metric:.4f} is not lower than worst {worst_metric:.4f}. Skipping save.")



###############################################################################################
### Example Calls

## For Dice (where a higher Dice is better):     
# Filename: "dice0.6789_epoch0001.pth.tar"
# save_checkpoint(state, 
#                 save_dir='models', 
#                 filename=f"dice{dice:.4f}_epoch{epoch:04d}.pth.tar", 
#                 max_model_num=10,
#                 metric_key='best_val_dice',
#                 metric_regex_prefix=r'dice',
#                 maximize=True,
#                 DEBUG=True)

## For Total Validation Loss (where a lower loss is better):
# # Filename: "val_loss_tot1.2345e-02_epoch0001.pth.tar"
# save_checkpoint(state, 
#                 save_dir='models', 
#                 filename=f"val_loss_tot{loss_tot:.4e}_epoch{epoch:04d}.pth.tar", 
#                 max_model_num=10,
#                 metric_key='best_val_loss_tot',
#                 metric_regex_prefix=r'val_loss_tot',
#                 maximize=False,
#                 DEBUG=True)

###############################################################################################