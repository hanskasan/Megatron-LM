#!/bin/bash

# Runs the "1.7B" parameter model

MEGATRON_HOME="/home/lustre/NLP/Megatron-LM_clean"
DATASET_HOME="/home/lustre/datasets"

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

VOCAB_FILE=${DATASET_HOME}/vocabs/gpt2-vocab.json
MERGE_FILE=${DATASET_HOME}/vocabs/gpt2-merges.txt
DATA_PATH=${DATASET_HOME}/Wikipedia-GPT/hfbpe_gpt_training_data_text_document

# train-iters is originally 500000

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --lr 0.00001 \
    --train-iters 50000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1e-6 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --use-mcore-models \
    --transformer-impl local \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --overlap-grad-reduce \
"

    # --fp16 \
    # --attention-softmax-in-fp32 \

    # --seed 0 \
    # --loss-scale 1.0 \


    # --recompute-granularity full \
    # --recompute-method uniform \
    # --recompute-num-layers 4

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

python /home/lustre/NLP/Megatron-LM_clean/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load $CHECKPOINT_PATH

    # --save $CHECKPOINT_PATH \
