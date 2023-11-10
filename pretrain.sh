#!/bin/bash

#SBATCH --job-name=pt-run
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2

#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=18

ACCELERATOR="gpu"
DEVICES=2
NODES=1

TRAIN_PATH="../scratch/"
VALID_PATH="../scratch/test"
SIZE=512
BATCH_SIZE=10
MASKS=False
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

echo `date`

srun python3 train.py --train_path $TRAIN_PATH --valid_path $VALID_PATH --test_path $VALID_PATH --batch_size $BATCH_SIZE \
    --no-masks --step $STEP --seq_length $SEQ_LENGTH --freeze $FREEZE --size $SIZE --backbone $BACKBONE \
    --lr $LR --layers $LAYERS --accelerator $ACCELERATOR --devices $DEVICES --num_nodes $NODES --profiler simple #--checkpoint $CHECKPOINT
    #--testing

echo `date`