
"""
Function list:
    my_set_seed
    set_seed
    worker_init_fn
    
    Note:
        both set_seed and worker_init_fn are from https://github.com/BailiangJ/rethink-reg/blob/main/utils/__init__.py
        
        I will use my_set_seed instead
                Currently using cudnn deterministic really slows down the training
                Also I use num of work = 0, so currently no need for worker_init_fn
        
"""


###########################################################################
### ref: https://pytorch.org/docs/stable/notes/randomness.html
###########################################################################

### set seed for torch, numpy, python, and make cudnn deterministic
# def set_random_seed(seed):
#     # seed the torch's RNG for all devices (both CPU and CUDA)
#     torch.manual_seed(rand_seed)
#     # set python seed 
#     random.seed(rand_seed)
#     # set numpy seed 
#     np.random.seed(rand_seed)

#     torch.use_deterministic_algorithms(True)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# ### handle seed for dataloader
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     numpy.random.seed(worker_seed)
#     random.seed(worker_seed)

# rand_seed = 2024
# set_random_seed(rand_seed)
# g = torch.Generator()
# g.manual_seed(0)

# DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     worker_init_fn=seed_worker,
#     generator=g,
# )
###########################################################################

import os
import random
import numpy as np
import torch

def my_set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    ### remove due to speed
    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'!!! IMPORTANT!!! Random seed set as {seed} in utils_rand_seed.my_set_seed()')


def set_seed(seed: int = 42) -> None:
    """
    From https://github.com/BailiangJ/rethink-reg/blob/6fc0af1f04a707bddbcfb5246e09e295d0b3a8fe/utils/__init__.py#L12
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'Random seed set as {seed}')


def worker_init_fn(worker_id):
    """Check https://github.com/Project-MONAI/MONAI/issues/1068."""
    worker_info = torch.utils.data.get_worker_info()
    try:
        worker_info.dataset.transform.set_random_state(worker_info.seed %
                                                       (2 ** 32))
    except AttributeError:
        pass
