#!/bin/bash

### NEVER COMMENT THIS OUT! ###
export BASE_PATH=/usr/lib/x86_64-linux-gnu/
export NCCL_ALGO='Ring'
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

### ADDITIONAL CONFIGURATIONS ###
# export NCCL_MAX_NCHANNELS='1'


### RUN! ###
export LD_LIBRARY_PATH=/home/lustre/libs/baseline/nccl/build/lib/:$BASE_PATH
/home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/test_TE.report
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/baseline_8GPUs.report
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/overlap-baseline_8GPUs.report
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/top1_8GPUs.report


# export NCCL_MIN_NCHANNELS='24'
# export NCCL_MAX_NCHANNELS='24'
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/baseline_8GPUs_${NCCL_MAX_NCHANNELS}chan.report
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/top1_8GPUs_${NCCL_MAX_NCHANNELS}chan.report

# export NCCL_MIN_NCHANNELS='12'
# export NCCL_MAX_NCHANNELS='12'
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/baseline_8GPUs_${NCCL_MAX_NCHANNELS}chan.report
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/top1_8GPUs_${NCCL_MAX_NCHANNELS}chan.report

# export NCCL_MIN_NCHANNELS='6'
# export NCCL_MAX_NCHANNELS='6'
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/baseline_8GPUs_${NCCL_MAX_NCHANNELS}chan.report
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/top1_8GPUs_${NCCL_MAX_NCHANNELS}chan.report


# export LD_LIBRARY_PATH=/home/lustre/libs/allreduce/shifting_skipreduce/random/build/lib/:$BASE_PATH
# export NCCL_SKIP_RS='4'
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/overlap-randomskip-${NCCL_SKIP_RS}RS_8GPUs.report
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/test_normal_skip-4RS.report
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/test_TE_skip-4RS.report

# export NCCL_MIN_NCHANNELS='24'
# export NCCL_MAX_NCHANNELS='24'
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/randomskip-${NCCL_SKIP_RS}RS_8GPUs_${NCCL_MAX_NCHANNELS}chan.report

# export NCCL_MIN_NCHANNELS='12'
# export NCCL_MAX_NCHANNELS='12'
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/randomskip-${NCCL_SKIP_RS}RS_8GPUs_${NCCL_MAX_NCHANNELS}chan.report

# export NCCL_MIN_NCHANNELS='6'
# export NCCL_MAX_NCHANNELS='6'
# /home/lustre/NLP/Megatron-LM_clean/scripts/timing_gpt_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_gpt_dp/randomskip-${NCCL_SKIP_RS}RS_8GPUs_${NCCL_MAX_NCHANNELS}chan.report
