#!/bin/bash

### NEVER COMMENT THIS OUT! ###
export BASE_PATH=/usr/lib/x86_64-linux-gnu/
export NCCL_ALGO='Ring'

### ADDITIONAL CONFIGURATIONS ###
# export NCCL_MAX_NCHANNELS='1'

### CHOOSE YOUR LIBRARY ###
export LD_LIBRARY_PATH=/home/lustre/libs/baseline/nccl/build/lib/:$BASE_PATH

# export LD_LIBRARY_PATH=/home/lustre/libs/allreduce/shifting_skipreduce/random/build/lib/:$BASE_PATH
# export NCCL_SKIP_RS='4'

### RUN! ###
/home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_interactive.sh
