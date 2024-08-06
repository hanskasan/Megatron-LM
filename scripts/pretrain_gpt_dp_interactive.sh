#!/bin/bash

# Runs the "1.7B" parameter model

MEGATRON_HOME="/home/lustre/NLP/Megatron-LM_clean"
DATASET_HOME="/home/lustre/datasets"

export CUDA_DEVICE_MAX_CONNECTIONS=1

# LIBRARIES
export BASE_PATH=/usr/lib/x86_64-linux-gnu/
# export LD_LIBRARY_PATH=/home/lustre/libs/allreduce/pruning/random/build/lib/:$BASE_PATH
export LD_LIBRARY_PATH=/home/lustre/libs/allreduce/shifting_skipreduce/random_oscillate/build/lib/:$BASE_PATH

# export NCCL_PROB_NUM_1='2'
# export NCCL_PROB_DENUM_1='4'
# export NCCL_PROB_NUM_2='2'
# export NCCL_PROB_DENUM_2='4'
# export NCCL_WARMUP_PERIOD='1'

export NCCL_OSC_PERIOD='1'
export NCCL_SKIP_RS_1='0'
export NCCL_SKIP_RS_2='6'

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=${MEGATRON_HOME}/checkpoints/pretrain_gpt1.7B_dp/baseline/
VOCAB_FILE=${DATASET_HOME}/vocabs/gpt2-vocab.json
MERGE_FILE=${DATASET_HOME}/vocabs/gpt2-merges.txt
DATA_PATH=${DATASET_HOME}/Wikipedia-GPT/hfbpe_gpt_training_data_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# train-iters is originally 500000

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 2304 \
    --num-attention-heads 24 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --lr 0.00001 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1e-6 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --fp16 \
    --use-mcore-models \
    --transformer-impl local \
    --overlap-grad-reduce \
"

    # --attention-softmax-in-fp32 \
    # --loss-scale 1.0 \
    # --recompute-granularity full \
    # --recompute-method uniform \
    # --recompute-num-layers 4
    # --ddp-bucket-size 43000000 \


DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS /home/lustre/NLP/Megatron-LM_clean/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --local-clip-grad 0.0 \
    --measure-aot \
    # --load $CHECKPOINT_PATH \
    # --save $CHECKPOINT_PATH \
    # --overlap-grad-reduce
