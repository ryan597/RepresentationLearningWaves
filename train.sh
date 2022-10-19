#!bin/bash

#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --account ndear024a
#SBATCH --partition GpuQ
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

BATCH_SIZE=5
MASKS=True
STEP=1
SEQ_LENGTH=3
FREEZE=0
SIZE=512
LR=0.0001
LAYERS=50
CHECKPOINT="outputs/lightning_logs/version/"

echo "--train_path $TRAIN_PATH --valid_path $VALID_PATH --test_path $VALID_PATH --batch_size $BATCH_SIZE \
    --masks $MASKS --step $STEP --seq_length $SEQ_LENGTH --freeze $FREEZE --size $size \
    --lr $LR --layers $LAYERS --checkpoint $CHECKPOINT"

python3 train.py --train_path $TRAIN_PATH --valid_path $VALID_PATH --test_path $VALID_PATH --batch_size $BATCH_SIZE \
    --masks $MASKS --step $STEP --seq_length $SEQ_LENGTH --freeze $FREEZE --size $size \
    --lr $LR --layers $LAYERS --checkpoint $CHECKPOINT
