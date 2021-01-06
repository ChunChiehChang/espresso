#!/bin/bash

# This script functions as a wrapper of a bash command that uses GPUs.
#
set -e

. ./path.sh

CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu

export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | sed "s=,$==g")

echo "$0: Running the job on GPU(s) $CUDA_VISIBLE_DEVICES"
"$@"
