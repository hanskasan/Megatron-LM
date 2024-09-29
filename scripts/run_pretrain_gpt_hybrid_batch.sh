#!/bin/bash

### NEVER COMMENT THIS OUT! ###
export BASE_PATH=/usr/lib/x86_64-linux-gnu/
export NCCL_ALGO='Ring'
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

### ADDITIONAL CONFIGURATIONS ###
# export NCCL_MAX_NCHANNELS='1'

### RUN! ###
# export LD_LIBRARY_PATH=/home/lustre/libs/baseline/nccl/build/lib/:$BASE_PATH
# /home/lustre/NLP/Megatron-LM_clean/scripts/pretrain_gpt_hybrid_batch.sh > /home/lustre/NLP/Megatron-LM_clean/aot_reports/pretrain_gpt_hybrid/baseline.report


export LD_LIBRARY_PATH=/home/lustre/libs/allreduce/shifting_skipreduce/random_oscillate/build/lib/:$BASE_PATH
export NCCL_WARMUP_PERIOD='0'
export NCCL_COOLDOWN_START='2000000'
export NCCL_OSC_PERIOD='1'
export NCCL_SKIP_RS_1='0'
export NCCL_SKIP_RS_2='6'
export NCCL_PROTECT_FROM='-1'
export NCCL_PROTECT_TO='-1'
/home/lustre/NLP/Megatron-LM_clean/scripts/pretrain_gpt_hybrid_batch.sh > /home/lustre/NLP/Megatron-LM_clean/aot_reports/pretrain_gpt_hybrid/oscillate.report
