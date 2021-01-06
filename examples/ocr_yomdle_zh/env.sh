#!/bin/bash

module load cuda10.1/toolkit || exit 1
module load gcc/7.2.0
module load cuda10.1/blas/10.1.105
module load cuda10.1/toolkit/10.1.105
module load cudnn/7.6.3_cuda10.1
module load intel/mkl/64/2019/5.281
module load nccl/2.4.7_cuda10.1

PS1="[\d \t] \u@\h:\w\\$ \[$(tput sgr0)\]"
. /opt/anaconda2/etc/profile.d/conda.sh
conda deactivate
source activate espresso_env
