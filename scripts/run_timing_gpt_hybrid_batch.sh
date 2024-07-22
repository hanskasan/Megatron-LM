#!/bin/bash

### NEVER COMMENT THIS OUT! ###
export BASE_PATH=/usr/lib/x86_64-linux-gnu/
export NCCL_ALGO='Ring'
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

### ADDITIONAL CONFIGURATIONS ###
# export NCCL_MAX_NCHANNELS='1'

### RUN! ###
export LD_LIBRARY_PATH=/home/lustre/libs/baseline/nccl/build/lib/:$BASE_PATH
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_hybrid_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_hybrid/baseline_16GPUs.report
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_hybrid_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_hybrid/baseline-overlap_16GPUs.report
/home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_hybrid_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_hybrid/test.report

# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_hybrid_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_hybrid/verysmall-overlap_16GPUs.report
