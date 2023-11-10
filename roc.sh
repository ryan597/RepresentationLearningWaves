#!/bin/bash

#SBATCH --job-name=roc
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18

ACCELERATOR="gpu"
DEVICES=1
NODES=1

TRAIN_PATH="../scratch/"
SIZE=512
BATCH_SIZE=10
MASKS=False

BACKBONE=$1
STEP=$2
SEQ_LENGTH=$3
CHECKPOINT=$4

echo step: $STEP
echo seq_length: $SEQ_LENGTH
echo backbone: $BACKBONE
echo checkpoint: $CHECKPOINT

source ~/.bashrc
conda activate rlwave

srun python3 get_auc.py --train_path $TRAIN_PATH --batch_size $BATCH_SIZE \
    --step $STEP --seq_length $SEQ_LENGTH --size $SIZE --backbone $BACKBONE \
    --checkpoint $CHECKPOINT
