#!/bin/bash

#SBATCH --job-name=RLW
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH  --cpus-per-task=18

ACCELERATOR="gpu"
DEVICES=1
NODES=1

TRAIN_PATH="../scratch/"
VALID_PATH="../scratch/test"
SIZE=256
BATCH_SIZE=16
MASKS=True
LR=0.001

BACKBONE=$1
STEP=$2
SEQ_LENGTH=$3
FREEZE=$4
LAYERS=50
CHECKPOINT=$5

echo step: $STEP
echo seq_length: $SEQ_LENGTH
echo backbone: $BACKBONE
echo freeze: $FREEZE
echo checkpoint: $CHECKPOINT

source ~/.bashrc
conda activate rlwave

srun python3 train.py --train_path $TRAIN_PATH --valid_path $VALID_PATH --test_path $VALID_PATH --batch_size $BATCH_SIZE \
    --masks --step $STEP --seq_length $SEQ_LENGTH --freeze $FREEZE --size $SIZE --backbone $BACKBONE \
    --lr $LR --layers $LAYERS --accelerator $ACCELERATOR --devices $DEVICES --num_nodes $NODES --checkpoint $CHECKPOINT
    #--testing
