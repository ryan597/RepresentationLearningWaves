#!/bin/sh 

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -A ndmat033a
#SBATCH -p GpuQ

module load intel/2019
module load cuda/11.3
module load cudnn
module load conda/2
source activate rlwave

MAX_EPOCHS=100
ACCELERATOR="gpu"
DEVICES=2
STRATEGY="ddp"
NUM_NODES=1
VAL_EPOCHS=2
DEFAULT_DIR="outputs/"
TRAIN_PATH="data/v1"
VALID_PATH="data/v2"

python3 lightmodel.py  --train_path $TRAIN_PATH --valid_path $VALID_PATH --enable_checkpointing True \
     --enable_progress_bar True --max_epochs $MAX_EPOCHS --accelerator $ACCELERATOR --devices $DEVICES \
     --strategy $STRATEGY --num_nodes $NUM_NODES --sync_batchnorm True  \
     --check_val_every_n_epoch $VAL_EPOCHS --logger True --default_root_dir $DEFAULT_DIR \
     --deterministic True --auto_lr_find True --auto_select_gpus True

