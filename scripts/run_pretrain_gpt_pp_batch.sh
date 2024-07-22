#!/bin/bash

# THERE SHOULD BE NOTHING TO CHANGE HERE! SIT BACK AND RELAX

### NEVER COMMENT THIS OUT! ###
export NCCL_ALGO='Ring'

export LD_LIBRARY_PATH=$NCCL_PATH

### RUN! ###
/home/lustre/NLP/Megatron-LM_clean/scripts/pretrain_gpt_pp_batch.sh >> $REPORT_PATH