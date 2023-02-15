#!/bin/bash

#SBATCH --job-name=RLW
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --account=ndear024a
#SBATCH --partition=GpuQ
#SBATCH --ntasks-per-node=2
#SBATCH --exclude=n[363,364,368]

ACCELERATOR="gpu"
DEVICES=2
NODES=1

TRAIN_PATH="data"
VALID_PATH="data/test"
SIZE=512
BATCH_SIZE=4
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


module purge
module load intel/2019u5
module load cuda/11.3
module load cudnn
module load conda/2
source activate rlwave

export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

srun python3 train.py --train_path $TRAIN_PATH --valid_path $VALID_PATH --test_path $VALID_PATH --batch_size $BATCH_SIZE \
    --masks $MASKS --step $STEP --seq_length $SEQ_LENGTH --freeze $FREEZE --size $SIZE --backbone $BACKBONE \
    --lr $LR --layers $LAYERS --accelerator $ACCELERATOR --devices $DEVICES --num_nodes $NODES #--checkpoint $CHECKPOINT
    #--testing True
