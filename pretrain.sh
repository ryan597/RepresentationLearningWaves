#!/bin/sh

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH -A ndmat033a
#SBATCH -p GpuQ

#SBATCH --mail-user=ryan.smith@ucdconnect.ie
# --mail-type=BEGIN

module load intel/2019
module load cuda/11.3
module load cudnn
module load conda/2
source activate rlwave

# NCCL settings
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=4

# Model inputs
MASKS=False
BACKBONE=resnet #baseline
LAYERS=50
DUAL=True
FREEZE=5
SIZE=128

# HParams
MAX_EPOCHS=100
LR=0.001
BATCH_SIZE=40
STRATEGY="ddp_find_unused_parameters_false"
ACCELERATOR="gpu"
DEVICES=2
NUM_NODES=1
VAL_EPOCHS=100
DEFAULT_DIR="outputs/"
TRAIN_PATH="data"
VALID_PATH="data/test"
LOG_EVERY_N_STEPS=1
NUM_SANITY_STEPS=0
GRADIENT_CLIP=0.5
#CHECKPOINT="outputs/lightning_logs/version_962084/checkpoints/epoch=114-step=1035.ckpt"


srun python3 train.py --masks $MASKS --backbone $BACKBONE --lr $LR --dual $DUAL --layers $LAYERS --freeze $FREEZE \
     --batch_size $BATCH_SIZE --train_path $TRAIN_PATH --valid_path $VALID_PATH --size $SIZE --enable_checkpointing True \
     --enable_progress_bar True --max_epochs $MAX_EPOCHS --strategy $STRATEGY --devices $DEVICES \
     --accelerator $ACCELERATOR --num_nodes $NUM_NODES --sync_batchnorm True  \
     --check_val_every_n_epoch $VAL_EPOCHS --logger True --default_root_dir $DEFAULT_DIR \
     --auto_lr_find True --auto_select_gpus True --num_sanity_val_steps $NUM_SANITY_STEPS \
     --log_every_n_steps $LOG_EVERY_N_STEPS --gradient_clip_val $GRADIENT_CLIP \
 #    --checkpoint $CHECKPOINT

