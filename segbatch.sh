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

python3 train.py -c train_seg

