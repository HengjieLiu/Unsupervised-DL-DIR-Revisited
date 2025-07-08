
"""
NOTES:
1. torch.backends
    Currently I am using "torch.backends.cudnn.benchmark" instead of "torch.backends.cudnn.deterministic". 
    Using the later would better ensure reproducibility but significantly decrease speed. My experience is that
    using "torch.backends.cudnn.benchmark" still result in very reproducible results for DL-DIR.
   

"""

######################################################################################################################
###### Imports (Preliminary)
######################################################################################################################
import os
import argparse


######################################################################################################################
###### Parse the commandline
###################################################################################################################### 
parser = argparse.ArgumentParser()

###### dataset
dataset_choices = [
    'OASIS_v1', 'LUMIR_v1', 
    'L2R20Task3_AbdominalCT_v1', 'L2R20Task3_AbdominalCT_v2',
]
parser.add_argument('--dataset-name', required=True, type=str, choices=dataset_choices, help='dataset name')
# args.dataset_name


###### model
parser.add_argument('--model-type', required=True, type=str, help='model type')
# choices=['bs1_vxm_v0_2', 'bs2_tm_v0', 'bs3_vfa_v0', 'bs4_sitreg_v0', 'fedd', 'fedd-bir']
# args.model_type

### fedd models specifics
parser.add_argument('--dir-config', required=False, type=str, default='standard', help='folder that contain config files for fedd models')
# args.dir_config
parser.add_argument('--use-batch-parallel', action='store_true', help='use use_batch_parallel for faster feature encoding') # store_true means without sepcifying default=False
## Note: this is handled differently for fedd-bidir due to my current implementation of sitreg models
# args.use_batch_parallel


###### trn_mode
parser.add_argument('--trn-mode-1', required=False, type=str, default='bidir', choices=['onepass', 'bidir'], help='training mode (default: standard)')
# Note1: choices=['onepass', 'bidir']  # "ic" (inverse_consistency) is deprecated; may add "gc" (group_consistency) later
# Note2: we need to distinguish between bidir_model and bidir_trn_mode
# args.trn_mode_1


###### loss
parser.add_argument('--ncc-type', required=False, type=str, default='vfa', help='Which NCC loss to use')
# args.ncc_type
parser.add_argument('--dice-type', type=str, default=None, help='Which Dice loss to use')
# args.dice_type


###### Hyperparameters
### learning rate and scheduling
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
# args.lr
# parser.add_argument('--lr-schedule', action='store_true', help='use lr scheduling') # store_true means without sepcifying default=False
parser.add_argument('--lr_schedule', type=str, default="none", choices=["none", "power", "cosine"],
                    help="Select scheduler type: 'none' for fixed lr, 'power' for your implemented scheduler, or 'cosine' for CosineAnnealingWarmRestarts")
# args.lr_schedule
## parameters for CosineAnnealingWarmRestarts
parser.add_argument('--T_0', type=int, default=100, help='Initial restart period for CosineAnnealingWarmRestarts')
parser.add_argument('--T_mult', type=int, default=1, help='Multiplier for T_0 for CosineAnnealingWarmRestarts')
parser.add_argument('--eta_min', type=float, default=1e-5, help='Minimum learning rate for CosineAnnealingWarmRestarts')

### weights
parser.add_argument('--lambda', required=True, type=float, dest='weight_lambda', default=0.01, \
    help='weight of deformation loss (default: 0.01)')
# args.weight_lambda
parser.add_argument('--gamma', required=False, type=float, dest='weight_gamma', default=None, \
    help='weight of dice loss (default: None [dice loss is not used])')
# args.weight_gamma


###### Training basics
parser.add_argument('--fp16', action='store_true', help='use fp16 training') # store_true means without sepcifying default=False
# args.fp16
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
# args.gpu: ['0', '1', '2', '3']
parser.add_argument('--subset-size', type=int, default=None, help='use a subset of data from all training samples (default: None), different for datasets')
# parser.add_argument('--subset-size', type=int, default=300, help='use a subset of data from all training samples (default: 300)')
# args.subset_size
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
# args.batch_size
parser.add_argument('--epochs', type=int, default=500, help='number of training epochs (default: 500)')
# args.epochs
parser.add_argument('--steps-per-epoch', type=int, default=None, help='number of training samples/pairs per epoch (default: None)')
# args.steps_per_epoch
parser.add_argument('--epoch-save', type=int, default=10, help='frequency of model saves (default: 10)')
# args.epoch_save
parser.add_argument('--epoch-val', type=int, default=10, help='frequency of validation (default: 10)')
# args.epoch_val
parser.add_argument('--epoch-early-check', type=int, default=100, help='early save the models at 100 (default: 100)')
# args.epoch_early_check

### Continue training with checkpoint loading
parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to a checkpoint file to continue training')
# args.checkpoint_path

parser.add_argument('--pretext-path', type=str, default=None, help='Path to a pretext checkpoint file initialize feature encoder')
# args.pretext_path
parser.add_argument('--no-freeze', action='store_true', help='unfreeze FeatureEncoder weights') # store_true means without sepcifying default=False
# args.no_freeze

###### Outputs
parser.add_argument('--max-model-num', type=int, required=False, default=10, help='max model checkpoints to save on disk')
# args.max_model_num
parser.add_argument('--out-dir', type=str, required=True, help='base output directory for models and logs')
# args.out_dir
parser.add_argument('--run-name', type=str, required=True, help='output folder for models and logs')
# args.run_name
parser.add_argument('--out-wandb', action='store_true', help='log with wandb') # store_true means without sepcifying default=False
# args.out_wandb


###### Setups related to dataloading and training speed 
parser.add_argument('--num-workers', type=int, default=0, help='num_workers (default: 0)')
# args.num_workers
parser.add_argument('--cudnn', choices=['det', 'ben', 'default'], required=True, 
    help='Set the mode for cudnn (deterministic, benchmark, or default)')
# args.cudnn


###### rand_seed
parser.add_argument('--rand-seed', type=int, required=False, default=None, help='Set the random seed to make training data loading deterministic')
# args.rand_seed

###### DEBUG FLAGS
parser.add_argument('--DEBUG-TRAIN-LOADER', action='store_true', help='debug train loader and random seed') # store_true means without sepcifying default=False
# args.DEBUG_TRAIN_LOADER
parser.add_argument('--DEBUG-TRAIN-MODE-1', action='store_true', help='debug train mode 1 (bidir related: bidir_model and bidir_trn_mode)') # store_true means without sepcifying default=False
# args.DEBUG_TRAIN_MODE_1
parser.add_argument('--DEBUG-CHECKPOINT', action='store_true', help='debug checkpoint saving') # store_true means without sepcifying default=False
# args.DEBUG_CHECKPOINT

args = parser.parse_args()
######################################################################################################################


### Get dataset_name and import associated packages
os.environ["DATASET_NAME"] = args.dataset_name
from common_imports_all_v1 import *

### DEBUGGING FLAGS
DEBUG_TRAIN_LOADER = args.DEBUG_TRAIN_LOADER ### Check the random seed settings and its impact on reproducible train loader
DEBUG_TRAIN_MODE_1= args.DEBUG_TRAIN_MODE_1 ### Check bidir related: bidir_model and bidir_trn_mode
DEBUG_CHECKPOINT = args.DEBUG_CHECKPOINT

### Set random seed!!!
if args.rand_seed is not None:
    utils_rand_seed.my_set_seed(args.rand_seed)


######################################################################################################################
### Local functions from TransMorph standard training
######################################################################################################################
def adjust_learning_rate_power(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    ### HJ modified
    lr_new = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)
    print(f"epoch: {epoch}, orginal lr {INIT_LR:.4e}, updated lr {lr_new:.4e}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new



######################################################################################################################
###### Setups for outputs
######################################################################################################################
###### Continue training not handled for now
if args.checkpoint_path is None: 
    # base directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    run_name = args.run_name
    
    import pytz
    import datetime
    pst_timezone = pytz.timezone('America/Los_Angeles')
    datetime_pst = datetime.datetime.now(pst_timezone).strftime("_%Y%m%d%H%M%S")
    custom_run_name = run_name + datetime_pst
    
    # run_name directory
    dir_out = os.path.join(args.out_dir, custom_run_name)
    os.makedirs(dir_out, exist_ok=True)
    # model/checkpoint directory
    
    dir_out_checkpoint_final = os.path.join(dir_out, 'checkpoint_final')
    os.makedirs(dir_out_checkpoint_final, exist_ok=True)
    dir_out_checkpoint_early = os.path.join(dir_out, 'checkpoint_' + str(args.epoch_early_check).zfill(4))
    os.makedirs(dir_out_checkpoint_early, exist_ok=True)
    
    
    # log file
    log_fname_1 = 'log_01_all.log'
    log_fname_2 = 'log_02_trn.log'
    log_fname_3 = 'log_03_val.log'
    log_path_1  = os.path.join(dir_out, log_fname_1)
    log_path_2  = os.path.join(dir_out, log_fname_2)
    log_path_3  = os.path.join(dir_out, log_fname_3)
    
else:
    pass

###### Initiate logging
logger = Logger_all(log_path_1)
logger.add_log_file('trn', log_path_2)
logger.add_log_file('val', log_path_3)
# Redirect stdout and stderr to the logger
sys.stdout = logger
sys.stderr = logger
print("Registration training logging initialized.")
print("This message goes to the primary log and the terminal.")
logger.write_to_additional('trn', "This is a training-specific message.")
logger.write_to_additional('val', "This is a validation-specific message.")


### Log info
# print(f"Training start for run_name: {run_name}")
# print(f"JUST CHECKING hyper_name_L: {hyper_name_L}")
print(f"""
OUTPUT INFO:
\targs.out_dir: {args.out_dir}
\tdir_out: {dir_out}
\tdir_out_checkpoint_final: {dir_out_checkpoint_final}
\tdir_out_checkpoint_early: {dir_out_checkpoint_early}
\tlog_path_1: {log_path_1}
\tlog_path_2: {log_path_2}
\tlog_path_3: {log_path_3}
""")

###### Set up wandb
if args.out_wandb:
    print('\nUsing wandb')
    import wandb
    
    wandb_run = wandb.init(project="xxx", entity="xxxxx", tags=[dataset_name], name=custom_run_name, config=args)

    print(f"wandb local saving directory: {wandb_run.dir}")
    print(f"wandb run id: {wandb_run.id}")
    print(f"wandb run name: {wandb_run.name}")
    print('\n')
else:
    print('\nwandb is not used\n')



######################################################################################################################
###### Setups for training
######################################################################################################################
print("TRAINING INFO:")

###### gpu device handling
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda') # will use the first available GPU

###### cudnn
# consider include it in args
if args.cudnn == 'det':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('\ttorch.backends.cudnn.deterministic = True [VoxelMorph say this is faster]')
elif args.cudnn == 'ben':
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print('\ttorch.backends.cudnn.benchmark = True')
elif args.cudnn == 'default':
    print(f"\ttorch.backends.cudnn default settings:")
    print(f"\ttorch.backends.cudnn.deterministic: {torch.backends.cudnn.deterministic}") # False
    print(f"\ttorch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}") # False
    

    
'''
Initialize model
'''
### get models
print('#' * 100)
print('#' * 100)
print('#' * 100)
if args.model_type.startswith('fedd'):
    if 'bidir' in args.model_type: # fedd-bidir
        bidir_model = True
        use_batch_parallel = True
    else:
        bidir_model = False
        use_batch_parallel=args.use_batch_parallel
    
    model = utils_models_v2_fedd.get_models_v2(
        img_size, args.dir_config, model_type=args.model_type, bidir=bidir_model, 
        use_batch_parallel=use_batch_parallel,
        path_pretask_model=args.pretext_path, no_freeze=args.no_freeze)
else:
    if args.model_type.startswith('bs1_vxm_v0'):
        dir_vxm = voxelmorph_models_path
        dir_tm = None
    elif args.model_type == 'bs2_tm_v0':
        dir_vxm = None
        dir_tm = transmorph_dir
    else:
        dir_vxm = None
        dir_tm = None

    model, model_category_info = utils_models_v2_baseline.get_models_v2(
        img_size, args.model_type, dir_vxm=dir_vxm, dir_tm=dir_tm, return_info=True)
    list_standard, list_standard_concat, list_sitreg = model_category_info

    if args.model_type in list_sitreg:
        bidir_model = True
    else:
        bidir_model = False

model.to(device)

### Initialize spatial transformation function
# Currently only used for bs4_sitreg_v0 in utils_models_v2_baseline.model_predict_v2
spatial_trans = utils_warp.SpatialTransformer(img_size).to(device)


### Initialize spatial transformation function for Dice calculation (follow TransMorph)
reg_model_nearest = tm_utils.register_model(img_size, 'nearest')
reg_model_nearest.cuda()
# reg_model_bilinear = tm_utils.register_model(img_size, 'bilinear')
# reg_model_bilinear.cuda()

print('#' * 100)

'''
Initialize datasets
'''
########## datasets
if dataset_name == "OASIS_v1":
    subset_size = 300 if args.subset_size is None else args.subset_size
    FLAG_EVAL_DICE = True
    n_val_labels = 35
    base_dir = os.path.dirname(train_dir)
    train_set = myDataset_OASIS_v1_json(base_dir=base_dir, json_path=json_path, stage='train', subset_size=args.subset_size, with_seg=False, DEBUG=DEBUG_TRAIN_LOADER)
    val_set   = myDataset_OASIS_v1_json(base_dir=base_dir, json_path=json_path, stage='validation', with_seg=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
elif dataset_name == "LUMIR_v1":
    # subset_size not implemented yet
    FLAG_EVAL_DICE = False
    # base_dir = os.path.dirname(train_dir)
    if args.subset_size is None:
        train_set = L2RLUMIRJSONDataset(base_dir=train_dir, json_path=json_path, stage='train')
        val_set = L2RLUMIRJSONDataset(base_dir=val_dir, json_path=json_path, stage='validation')
    else:
        train_set = L2RLUMIRJSONDataset_subset(base_dir=train_dir, json_path=json_path, stage='train', subset_size=args.subset_size)
        val_set = L2RLUMIRJSONDataset(base_dir=val_dir, json_path=json_path, stage='validation')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
elif dataset_name.startswith("L2R20Task3_AbdominalCT"):
    # Currently for both L2R20Task3_AbdominalCT_v1 and L2R20Task3_AbdominalCT_v2
    subset_size = 30 if args.subset_size is None else args.subset_size
    FLAG_EVAL_DICE = True
    n_val_labels = 13
    base_dir = train_dir
    train_set = myDataset_L2R20TASK3CT_v1_json(base_dir=base_dir, json_path=json_path, stage='train', subset_size=args.subset_size, with_seg=False, DEBUG=DEBUG_TRAIN_LOADER)
    val_set   = myDataset_L2R20TASK3CT_v1_json(base_dir=base_dir, json_path=json_path, stage='validation', with_seg=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
else:
    raise ValueError(f"Unrecognized dataset_name: {dataset_name}")

iter_train_loader = iter(train_loader)
sample_size = args.steps_per_epoch if args.steps_per_epoch is not None else 100

### print some information
print(f"len(train_set): {len(train_set)}")
print(f"len(train_loader): {len(train_loader)}")
print(f"len(iter_train_loader): {len(iter_train_loader)}")
print(f"Training sample_size (per epoch): {sample_size}")
print(f"len(val_set): {len(val_set)}")
print(f"len(val_loader): {len(val_loader)}")

if args.trn_mode_1 == 'onepass':
    bidir_trn_mode = False
elif args.trn_mode_1 == 'bidir':
    bidir_trn_mode = True
print(f"\tINFO: trn_mode_1: {args.trn_mode_1}")



'''
Initialize training
'''
###### Initialize training (continue training not implemented)
if args.checkpoint_path is None: # Initialize
    epoch_start = 0
    max_epoch = args.epochs
    
    updated_lr = args.lr
          
    # training loss
    list_epoch_loss = []
    list_epoch_loss_sim = []
    list_epoch_loss_reg = []
    list_epoch_loss_seg = []
    
    # validation dice
    if FLAG_EVAL_DICE:
        best_val_dice = 0
    
    # validation loss
    best_val_loss_tot = float('inf')
    best_val_loss_sim = float('inf')

else: # Load checkpoint if specified !!! THIS IS NOT DONE OR VERIFIED YET !!!

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    epoch_start = checkpoint['epoch']
    max_epoch = args.epochs
    
    if args.lr_schedule == 'power':
        ### ??? seems only needed in continual training ...
        updated_lr = round(args.lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_0,
            T_mult=args.T_mult,
            eta_min=args.eta_min
        )
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        updated_lr = args.lr
    ### currently missing implementation for CosineAnnealingWarmRestarts


### weights for loss functions
# Check if use dice loss (gamma was explicitly provided)
if args.weight_gamma is None:
    use_seg = False
    weights     = [1, args.weight_lambda]
    print(f"\tINFO: Loss = L_sim + L_reg (not using dice loss)")
    print(f"\tINFO: args.weight_lambda: {args.weight_lambda}")
else:
    use_seg = True
    weights     = [1, args.weight_lambda, args.weight_gamma]
    print(f"\tINFO: Loss = L_sim + L_reg + L_seg")
    print(f"\tINFO: args.weight_lambda: {args.weight_lambda}, args.weight_gamma: {args.weight_gamma}")

optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True) # I will for now keep amsgrad=True

## Instantiate the cosine scheduler if selected: Immediately after creating the optimizer, add
if args.lr_schedule == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.T_0,
        T_mult=args.T_mult,
        eta_min=args.eta_min
    )


if args.fp16:
    scaler = GradScaler()

if args.ncc_type is None or args.ncc_type == 'vxm':
    func_loss_sim = tm_losses.NCC_vxm()
elif args.ncc_type == 'vxm_fast':
    func_loss_sim = NCC_vxm_fast()
elif args.ncc_type == 'gauss':
    func_loss_sim = NCC_gauss()
elif args.ncc_type == 'vfa':
    func_loss_sim = NCC_vfa()
elif args.ncc_type == 'vfa_fast':
    func_loss_sim = NCC_vfa_fast()
else:
    raise NotImplementedError(f"NCC loss type '{args.ncc_type}' is not implemented.")
func_loss_reg = tm_losses.Grad3d(penalty='l2')
if use_seg:
    if args.dice_type is None:
        func_loss_seg = tm_losses.DiceLoss() # TransMorph/V-Net version
    elif args.dice_type == 'vxm':
        func_loss_seg = Dice_vxm() # VoxelMorph version
    else:
        raise NotImplementedError(f"Dice loss type '{args.ncc_type}' is not implemented.")
    func_losses = [func_loss_sim, func_loss_reg, func_loss_seg]
else:
    func_losses = [func_loss_sim, func_loss_reg]



for epoch in range(epoch_start, max_epoch):
    
    if DEBUG_CHECKPOINT:
        break
    
    ### modify saving path
    if epoch >= args.epoch_early_check:
        dir_out_checkpoint = dir_out_checkpoint_final
    else:
        dir_out_checkpoint = dir_out_checkpoint_early
    
    
    # At the start of an epoch, reset peak stats:
    torch.cuda.reset_peak_memory_stats()
    
    ### adjust_learning_rate_power
    if args.lr_schedule == "power":
        adjust_learning_rate_power(optimizer, epoch, max_epoch, args.lr)
    
    ################## training loop ##################
    ### Set to train
    model.train()
   
    ### Init for training info
    # time
    epoch_step_time = 0.0
    # loss
    epoch_loss      = 0.0
    epoch_loss_sim  = 0.0
    epoch_loss_reg  = 0.0


    # NEW: Determine effective_loader_steps based on training mode:
    # if bidir_model or (not bidir_model and bidir_trn_mode):
    #     effective_loader_steps = sample_size // 2
    # else:
    #     effective_loader_steps = sample_size
    # for trn_step in range(effective_loader_steps):
    # for trn_step in range(sample_size):
    # for trn_step, data in enumerate(train_loader):

    trn_step = 0  # This now tracks the number of sample updates performed
    while trn_step < sample_size:
    
        # ###$$$ DEBUG DEBUG_TRAIN_LOADER
        # if DEBUG_TRAIN_LOADER:
        #     time.sleep(1) # slow down dataloader to see the process
        #     continue
        
        step_start_time = time.time()
        
        # Get batch; reinitialize iterator if needed
        try:
            data = next(iter_train_loader)
        except StopIteration:
            # End of an epoch reached, reinitialize so that the DataLoader reshuffles the data
            iter_train_loader = iter(train_loader)
            data = next(iter_train_loader)


        ###$$$ DEBUG DEBUG_TRAIN_LOADER
        if DEBUG_TRAIN_LOADER:
            trn_step += 1
            print(f"trn_step/sample_size: {trn_step}/{sample_size}")
            epoch_step_time += (time.time() - step_start_time)
            # time.sleep(0.1) # slow down dataloader to see the process
            continue

        
        
        data = [t.cuda() for t in data]
        moving = data[0]
        fixed  = data[1]


        ############################## ---- Branch based on training mode ---- ##############################
        ############### Case 1: bidir model: bidir_model=True, no matter bidir_trn_mode
        # Currently only 2 cases: fedd-bidir, list_sitreg
        if bidir_model:

            if DEBUG_TRAIN_MODE_1:
                trn_step += 2
                print(f"bidir_model: {bidir_model}, bidir_trn_mode: {bidir_trn_mode}, trn_step/sample_size: {trn_step}/{sample_size}")
                epoch_step_time += (time.time() - step_start_time)
                continue
                
            with autocast(dtype=torch.float16) if args.fp16 else nullcontext(): # wrap the forward pass and loss calculation in this for mixed precision training
                if args.model_type.startswith('fedd-bidir'):
                    deformed_1, deformed_2, dvf_12, dvf_21 = utils_models_v2_fedd.model_predict_v2(moving, fixed, model, bidir=bidir_model)
                elif args.model_type in list_sitreg:
                    deformed_1, deformed_2, dvf_12, dvf_21 = utils_models_v2_baseline.model_predict_v2(moving, fixed, model, args.model_type, spatial_trans)
                else:
                    raise NotImplementedError("Unknown model type for bidir model.")
                    
                ### loss calculation
                image_1 = moving # can move to front
                image_2 = fixed  # can move to front
                loss_sim = (func_losses[0](image_2, deformed_1) + func_losses[0](image_1, deformed_2)) * weights[0] * 0.5
                loss_reg = (func_losses[1](dvf_12, 0) + func_losses[1](dvf_21, 0)) * weights[1] * 0.5
                loss = loss_sim + loss_reg

            ### backpropagate and optimize (don't put in with block)
            optimizer.zero_grad()
            
            if args.fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            ### update trn_step (and lr_schedule)
            # Inside your inner training loop (after optimizer.step()):
            if args.lr_schedule == "cosine":
                # This creates a fractional/smoother lr update
                scheduler.step(epoch + trn_step / sample_size)
            trn_step += 2  # Bidirectional model processes 2 samples per update
            
            ### record and log
            # Accumulate losses and time
            epoch_loss     += loss.item()
            epoch_loss_sim += loss_sim.item()
            epoch_loss_reg += loss_reg.item()
            epoch_step_time += (time.time() - step_start_time)
            
            

        ############### Case 2: non bidir model but bidir trn_mode: bidir_model=False, bidir_trn_mode=True
        elif not bidir_model and bidir_trn_mode:

            if DEBUG_TRAIN_MODE_1:
                trn_step += 1  # Count first update
                trn_step += 1  # Count second update
                print(f"bidir_model: {bidir_model}, bidir_trn_mode: {bidir_trn_mode}, trn_step/sample_size: {trn_step}/{sample_size}")
                epoch_step_time += (time.time() - step_start_time)
                continue
            
            ####### First update: (moving, fixed)
            moving_1 = moving
            fixed_1  = fixed
            with autocast(dtype=torch.float16) if args.fp16 else nullcontext(): # wrap the forward pass and loss calculation in this for mixed precision training
                if args.model_type.startswith('fedd'):
                    deformed, dvf = utils_models_v2_fedd.model_predict_v2(moving_1, fixed_1, model)
                elif args.model_type in list_standard or args.model_type in list_standard_concat:
                    deformed, dvf = utils_models_v2_baseline.model_predict_v2(moving_1, fixed_1, model, args.model_type)
                else:
                    raise NotImplementedError("Unknown model type for non bidir model but bidir trn_mode.")
    
                ### loss calculation
                loss_sim = func_losses[0](fixed_1, deformed) * weights[0]
                loss_reg = func_losses[1](dvf, 0) * weights[1]
                loss = loss_sim + loss_reg
            
            ### backpropagate and optimize (don't put in with block)
            optimizer.zero_grad()
            
            if args.fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            ### update trn_step (and lr_schedule)
            # Inside your inner training loop (after optimizer.step()):
            if args.lr_schedule == "cosine":
                # This creates a fractional/smoother lr update
                scheduler.step(epoch + trn_step / sample_size)
            trn_step += 1  # Count first update
            
            ### record and log
            # Accumulate losses and time
            epoch_loss     += loss.item()
            epoch_loss_sim += loss_sim.item()
            epoch_loss_reg += loss_reg.item()
            epoch_step_time += (time.time() - step_start_time)

            
            ####### Second update: (fixed, moving)
            step_start_time = time.time() # restart timer

            moving_2 = fixed
            fixed_2  = moving
            with autocast(dtype=torch.float16) if args.fp16 else nullcontext(): # wrap the forward pass and loss calculation in this for mixed precision training
                if args.model_type.startswith('fedd'):
                    deformed, dvf = utils_models_v2_fedd.model_predict_v2(moving_2, fixed_2, model)
                elif args.model_type in list_standard or args.model_type in list_standard_concat:
                    deformed, dvf = utils_models_v2_baseline.model_predict_v2(moving_2, fixed_2, model, args.model_type)
                else:
                    raise NotImplementedError("Unknown model type for non bidir model but bidir trn_mode.")
                    
                ### loss calculation
                loss_sim = func_losses[0](fixed_2, deformed) * weights[0]
                loss_reg = func_losses[1](dvf, 0) * weights[1]
                loss = loss_sim + loss_reg
            
            ### backpropagate and optimize (don't put in with block)
            optimizer.zero_grad()
            
            if args.fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            ### update trn_step (and lr_schedule)
            # Inside your inner training loop (after optimizer.step()):
            if args.lr_schedule == "cosine":
                # This creates a fractional/smoother lr update
                scheduler.step(epoch + trn_step / sample_size)
            trn_step += 1  # Count second update
            
            ### record and log
            # Accumulate losses and time
            epoch_loss     += loss.item()
            epoch_loss_sim += loss_sim.item()
            epoch_loss_reg += loss_reg.item()
            epoch_step_time += (time.time() - step_start_time)
        
        ############### Case 3: non bidir model and not bidir trn_mode: bidir_model=False, bidir_trn_mode=False
        else: # elif not bidir_model and not bidir_trn_mode:

            if DEBUG_TRAIN_MODE_1:
                trn_step += 1  # Only 1 pair
                print(f"bidir_model: {bidir_model}, bidir_trn_mode: {bidir_trn_mode}, trn_step/sample_size: {trn_step}/{sample_size}")
                epoch_step_time += (time.time() - step_start_time)
                continue
            
            ####### First and the only update: (moving, fixed)
            moving_1 = moving
            fixed_1  = fixed
            with autocast(dtype=torch.float16) if args.fp16 else nullcontext(): # wrap the forward pass and loss calculation in this for mixed precision training
                if args.model_type.startswith('fedd'):
                    deformed, dvf = utils_models_v2_fedd.model_predict_v2(moving_1, fixed_1, model)
                elif args.model_type in list_standard or args.model_type in list_standard_concat:
                    deformed, dvf = utils_models_v2_baseline.model_predict_v2(moving_1, fixed_1, model, args.model_type)
                else:
                    raise NotImplementedError("Unknown model type for non bidir model but bidir trn_mode.")
    
                ### loss calculation
                loss_sim = func_losses[0](fixed_1, deformed) * weights[0]
                loss_reg = func_losses[1](dvf, 0) * weights[1]
                loss = loss_sim + loss_reg
            
            ### backpropagate and optimize (don't put in with block)
            optimizer.zero_grad()
            
            if args.fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            ### update trn_step (and lr_schedule)
            # Inside your inner training loop (after optimizer.step()):
            if args.lr_schedule == "cosine":
                # This creates a fractional/smoother lr update
                scheduler.step(epoch + trn_step / sample_size)
            trn_step += 1  # One update
            
            ### record and log
            # Accumulate losses and time
            epoch_loss     += loss.item()
            epoch_loss_sim += loss_sim.item()
            epoch_loss_reg += loss_reg.item()
            epoch_step_time += (time.time() - step_start_time)

 
    ###$$$ DEBUG DEBUG_TRAIN_LOADER
    if DEBUG_TRAIN_LOADER:
        # continue
        epoch_step_time /= sample_size
        time_info  = f'{epoch_step_time:.4f} sec/step, {epoch_step_time*sample_size:.4f} sec/epoch'
        print(f"Done with 1 epoch for DEBUG_TRAIN_LOADER, time passed {time_info}")
        break

    if DEBUG_TRAIN_MODE_1:
        epoch_step_time /= sample_size
        time_info  = f'{epoch_step_time:.4f} sec/step, {epoch_step_time*sample_size:.4f} sec/epoch'
        print(f"Done with 1 epoch for DEBUG_TRAIN_MODE_1, time passed {time_info}")
        break
    
    ############ log training info 
    # Average losses and time over steps
    if bidir_model:
        epoch_loss /= (sample_size/2.)
        epoch_loss_sim /= (sample_size/2.)
        epoch_loss_reg /= (sample_size/2.)
        epoch_step_time /= sample_size # ? questionable
        # epoch_step_time /= (sample_size/2.) # ? questionable
    else:
        epoch_loss /= sample_size
        epoch_loss_sim /= sample_size
        epoch_loss_reg /= sample_size
        epoch_step_time /= sample_size
    
    # Store losses for history
    list_epoch_loss.append(epoch_loss)
    list_epoch_loss_sim.append(epoch_loss_sim)
    list_epoch_loss_reg.append(epoch_loss_reg)
    
    # Prepare logging information
    epoch_info = f'Epoch {epoch+1}/{max_epoch}'
    time_info  = f'{epoch_step_time:.4f} sec/step, {epoch_step_time*sample_size:.4f} sec/epoch'
    loss_info  = f'loss: {epoch_loss:.4e}, loss_sim: {epoch_loss_sim:.4e}, loss_reg: {epoch_loss_reg:.4e}'
    lr_curr = optimizer.param_groups[0]["lr"]
    lr_info = f'lr: {lr_curr:.4e}'

    trn_info = ' - '.join((epoch_info, time_info, loss_info, lr_info))
    
    ### log to training log
    logger.write_to_additional('trn', trn_info)

    ### log to wandb (training)
    if args.out_wandb:
        log_dict = {
            "train_loss_total": epoch_loss,
            "train_loss_sim": epoch_loss_sim,
            "train_loss_reg": epoch_loss_reg,
            "train_lr": lr_curr,
        }
        wandb.log(log_dict, step=epoch+1)

    # At end of training epoch: report memory usage
    utils_models_v2_baseline.report_gpu_memory()
    
    ################## END of training loop ##################
        
    
    
    ################## validation loop ##################
    if (epoch+1) % args.epoch_val == 0:
        
        ### Set to eval
        model.eval()
        
        
        with torch.no_grad():

            val_loss      = 0.0
            val_loss_sim  = 0.0
            val_loss_reg  = 0.0
            val_start_time = time.time()
            if FLAG_EVAL_DICE:
                list_val_dice = []
            
            for val_step, data in enumerate(val_loader):
                
                # if val_step>2:
                #     break
                
                data = [t.cuda() for t in data]

                if FLAG_EVAL_DICE:
                    moving, fixed, moving_seg, fixed_seg = data
                else:
                    moving, fixed = data
                
                
                if args.model_type.startswith('fedd'):
                    if not bidir_model:
                        deformed, dvf = utils_models_v2_fedd.model_predict_v2(moving, fixed, model)
    
                        ### loss calculation
                        loss_sim = func_losses[0](fixed, deformed) * weights[0]
                        loss_reg = func_losses[1](dvf, 0) * weights[1]

                    else:
                        deformed_1, deformed_2, dvf_12, dvf_21 = utils_models_v2_fedd.model_predict_v2(moving, fixed, model, bidir=bidir_model)
                    
                        ### loss calculation
                        image_1 = moving
                        image_2 = fixed
                        loss_sim = (func_losses[0](image_2, deformed_1) + func_losses[0](image_1, deformed_2)) * weights[0] * 0.5
                        loss_reg = (func_losses[1](dvf_12, 0) + func_losses[1](dvf_21, 0)) * weights[1] * 0.5
                        loss = loss_sim + loss_reg

                        ### assign forward dvf for dice calculation
                        dvf = dvf_12
                    
                elif args.model_type in list_standard or args.model_type in list_standard_concat:
                    deformed, dvf = utils_models_v2_baseline.model_predict_v2(moving, fixed, model, args.model_type)

                    ### loss calculation
                    loss_sim = func_losses[0](fixed, deformed) * weights[0]
                    loss_reg = func_losses[1](dvf, 0) * weights[1]

                elif args.model_type in list_sitreg:
                    deformed_1, deformed_2, dvf_12, dvf_21 = utils_models_v2_baseline.model_predict_v2(moving, fixed, model, args.model_type, spatial_trans)
                    
                    ### loss calculation
                    image_1 = moving
                    image_2 = fixed
                    loss_sim = (func_losses[0](image_2, deformed_1) + func_losses[0](image_1, deformed_2)) * weights[0] * 0.5
                    loss_reg = (func_losses[1](dvf_12, 0) + func_losses[1](dvf_21, 0)) * weights[1] * 0.5
                    loss = loss_sim + loss_reg
                    
                    ### assign forward dvf for dice calculation
                    dvf = dvf_12

                
                ### Calculate Dice
                
                if FLAG_EVAL_DICE:
                    deformed_seg  = reg_model_nearest([moving_seg.cuda().float(), dvf.cuda()])
                    # dice = tm_utils.dice_val_VOI(deformed_seg.long(), fixed_seg.long())
                    dice = utils_eval.dice_val_VOI(deformed_seg.long(), fixed_seg.long(), n_labels=n_val_labels)
                    val_dice = dice # dice is results from np.mean
                    list_val_dice.append(val_dice)
                
                
                ### record and log
                # Accumulate losses and time
                val_loss     += loss.item()
                val_loss_sim += loss_sim.item()
                val_loss_reg += loss_reg.item()
                
                

        ###### log training info
        # Average losses and time over steps

        if FLAG_EVAL_DICE:
            val_dice = np.mean(list_val_dice)
        val_loss /= len(val_loader)
        val_loss_sim /= len(val_loader)
        val_loss_reg /= len(val_loader)

        if FLAG_EVAL_DICE:
            best_val_dice = max(val_dice, best_val_dice)
        best_val_loss_tot = min(val_loss, best_val_loss_tot)
        best_val_loss_sim = min(val_loss_sim, best_val_loss_sim)

        # Organize val_info
        epoch_info = f'Val Epoch {epoch+1}/{max_epoch}'
        time_info = f'{time.time()-val_start_time:.2f} sec/validation'
        loss_info = f'loss: {val_loss:.4e}, loss_sim: {val_loss_sim:.4e}, loss_reg: {val_loss_reg:.4e}'
        best_loss_info = f'Best val_loss_tot: {best_val_loss_tot:.4f}, Best val_loss_sim: {best_val_loss_sim:.4f}'
        if FLAG_EVAL_DICE:
            dice_info = f'val_dice: {val_dice:.4f}'
            best_dice_info = f'Best dice: {best_val_dice:.4f}'
            val_info = ' - '.join((epoch_info, time_info, loss_info, dice_info, best_loss_info, best_dice_info))
        else:
            val_info = ' - '.join((epoch_info, time_info, loss_info, best_loss_info))
        
        ### log to validation log
        logger.write_to_additional('val', val_info)
        
        ### log to wandb (validation)
        if args.out_wandb:
            # Prepare wandb logging data
            wandb_data = {
                "val_loss": val_loss,
                "val_loss_sim": val_loss_sim,
                "val_loss_reg": val_loss_reg,
            }
            if FLAG_EVAL_DICE:
                wandb_data["val_dice"] = val_dice
                wandb_data["best_val_dice"] = best_val_dice
            # Log to wandb
            wandb.log(wandb_data, step=epoch+1) ###FLAG(epoch+1))
        
        ###### Save checkpoint
        state_checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss_tot': best_val_loss_tot,
            'best_val_loss_sim': best_val_loss_sim,
            **({'best_val_dice': best_val_dice} if FLAG_EVAL_DICE else {}),
            **({'scheduler': scheduler.state_dict()} if args.lr_schedule == "cosine" else {}),
        }
        # Determine checkpoint naming and metric parameters based on FLAG_EVAL_DICE
        if FLAG_EVAL_DICE:
            metric_key = 'best_val_dice'
            metric_regex_prefix = r'dice'
            filename_prefix = f"dice{val_dice:.4f}"
            maximize_metric = True
        else:
            metric_key = 'best_val_loss_sim'
            metric_regex_prefix = r'loss_sim'
            filename_prefix = f"loss_sim{val_loss_sim:.4f}"
            maximize_metric = False
        
        save_checkpoint(
            state_checkpoint, 
            save_dir=dir_out_checkpoint, 
            filename=f"{filename_prefix}_epoch{str(epoch+1).zfill(4)}.pth.tar", 
            max_model_num=args.max_model_num,
            metric_key=metric_key, 
            metric_regex_prefix=metric_regex_prefix, 
            maximize=maximize_metric, 
            DEBUG=DEBUG_CHECKPOINT
        )
        
    ################## END of validation loop ##################
    
    
### close logger
logger.close()