#!/bin/bash

### NEVER COMMENT THIS OUT! ###
export BASE_PATH=/usr/lib/x86_64-linux-gnu/
export NCCL_ALGO='Ring'
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

### RUN! ###
export LD_LIBRARY_PATH=/home/lustre/libs/baseline/nccl/build/lib/:$BASE_PATH
/home/lustre/NLP/Megatron-LM_clean/scripts/timing_vit_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_vit_dp/baseline.report

export LD_LIBRARY_PATH=/home/lustre/libs/allreduce/shifting_skipreduce/random/build/lib/:$BASE_PATH
export NCCL_SKIP_RS='4'
/home/lustre/NLP/Megatron-LM_clean/scripts/timing_vit_dp_batch.sh > /home/lustre/NLP/Megatron-LM_clean/timing_reports/pretrain_vit_dp/randomskip-${NCCL_SKIP_RS}RS.report