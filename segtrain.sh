#!/bin/bash

#SBATCH --job-name=RLW
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --account=ndear024a
#SBATCH --partition=GpuQ
#SBATCH --ntasks-per-node=2

module purge
module load intel/2019u5
module load cuda/11.3
module load cudnn
module load conda/2
source activate rlwave

export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

TRAIN_PATH="data"
VALID_PATH="data/test"
SIZE=512
BATCH_SIZE=5
MASKS=True
LR=0.0001


BACKBONE=$1
STEP=$2
SEQ_LENGTH=$3
FREEZE=$4
LAYERS=50

CHECKPOINT="outputs/lightning_logs/version_1013214/checkpoints/epoch=39-step=2360.ckpt" # step 1, seq_length 5, ATTN
#CHECKPOINT="outputs/lightning_logs/version_1013215/checkpoints/epoch=39-step=2360.ckpt"  # step 1, seq_length 3, ATTN
#CHECKPOINT="outputs/lightning_logs/version_1013216/checkpoints/epoch=39-step=2360.ckpt"  # step 1, seq_length 2, ATTN
#CHECKPOINT="outputs/lightning_logs/version_1013217/checkpoints/epoch=39-step=2360.ckpt"  # step 3, seq_length 5, ATTN
#CHECKPOINT="outputs/lightning_logs/version_1013218/checkpoints/epoch=39-step=2360.ckpt"  # step 3, seq_length 3, ATTN
#CHECKPOINT="outputs/lightning_logs/version_1013219/checkpoints/epoch=39-step=2360.ckpt"  # step 3, seq_length 2, ATTN
#CHECKPOINT="outputs/lightning_logs/version_1013220/checkpoints/epoch=39-step=2360.ckpt"  # step 5, seq_length 5, ATTN
#CHECKPOINT="outputs/lightning_logs/version_1013221/checkpoints/epoch=39-step=2360.ckpt"  # step 5, seq_length 3, ATTN
#CHECKPOINT="outputs/lightning_logs/version_1013224/checkpoints/epoch=39-step=2360.ckpt"  # step 10, seq_length 3, ATTN
#CHECKPOINT="outputs/lightning_logs/version_1013225/checkpoints/epoch=39-step=2360.ckpt"  # step 10, seq_length 2, ATTN

#CHECKPOINT="outputs/lightning_logs/version_1013219/checkpoints/epoch=39-step=2360.ckpt"  # step 3, seq_length 2, ATTN
#CHECKPOINT="outputs/lightning_logs/version_1013219/checkpoints/epoch=39-step=2360.ckpt"  # step 3, seq_length 2, ATTN



ACCELERATOR="gpu"
DEVICES=2
NODES=1

echo step: $STEP
echo seq_length: $SEQ_LENGTH
echo backbone: $BACKBONE
echo freeze: $FREEZE
echo checkpoint: $CHECKPOINT \n


srun python3 train.py --train_path $TRAIN_PATH --valid_path $VALID_PATH --test_path $VALID_PATH --batch_size $BATCH_SIZE \
    --masks $MASKS --step $STEP --seq_length $SEQ_LENGTH --freeze $FREEZE --size $SIZE --backbone $BACKBONE \
    --lr $LR --layers $LAYERS --accelerator $ACCELERATOR --devices $DEVICES --num_nodes $NODES --checkpoint $CHECKPOINT
    #--testing True
