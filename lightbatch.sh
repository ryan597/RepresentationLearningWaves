#!/bin/sh 

#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH -A ndmat033a
#SBATCH -p GpuQ

module load intel/2019
module load cuda/11.3
module load cudnn
module load conda/2
source activate rlwave

# Model inputs
MASKS=False
BACKBONE=resnet50

# HParams
MAX_EPOCHS=9000
ACCELERATOR="gpu"
DEVICES=2
STRATEGY="ddp"
NUM_NODES=2
VAL_EPOCHS=5
DEFAULT_DIR="outputs/"
TRAIN_PATH="data/v1"
VALID_PATH="data/v2"
LOG_EVERY_N_STEPS=1
NUM_SANITY_STEPS=0
GRADIENT_CLIP=0.5

python3 lightmodel.py --masks $MASKS --backbone $BACKBONE \
     --train_path $TRAIN_PATH --valid_path $VALID_PATH --enable_checkpointing True \
     --enable_progress_bar True --max_epochs $MAX_EPOCHS --accelerator $ACCELERATOR --devices $DEVICES \
     --strategy $STRATEGY --num_nodes $NUM_NODES --sync_batchnorm True  \
     --check_val_every_n_epoch $VAL_EPOCHS --logger True --default_root_dir $DEFAULT_DIR \
     --auto_lr_find True --auto_select_gpus True --num_sanity_val_steps $NUM_SANITY_STEPS \
     --log_every_n_steps $LOG_EVERY_N_STEPS --gradient_clip_val $GRADIENT_CLIP

