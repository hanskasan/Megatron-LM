#!/bin/bash

### NEVER COMMENT THIS OUT! ###
export BASE_PATH=/usr/lib/x86_64-linux-gnu/
export NCCL_ALGO='Ring'

### ADDITIONAL CONFIGURATIONS ###
# export NCCL_MAX_NCHANNELS='1'

### BASELINE ###
export LD_LIBRARY_PATH=/home/lustre/libs/baseline/nccl/build/lib/:$BASE_PATH

/home/lustre/NLP/Megatron-LM_clean/scripts/pretrain_gpt_mp_batch.sh >> /home/lustre/NLP/Megatron-LM_clean/reports/pretrain_mp/baseline_long.report