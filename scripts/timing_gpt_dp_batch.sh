#!/bin/bash

# Runs the "1.7B" parameter model

MEGATRON_HOME="/home/lustre/NLP/Megatron-LM_clean"
DATASET_HOME="/home/lustre/datasets/"

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8

CHECKPOINT_PATH=${MEGATRON_HOME}/checkpoints/pretrain_gpt_dp/
VOCAB_FILE=${DATASET_HOME}/vocabs/gpt2-vocab.json
MERGE_FILE=${DATASET_HOME}/vocabs/gpt2-merges.txt
DATA_PATH=${DATASET_HOME}/Wikipedia-GPT/hfbpe_gpt_training_data_text_document

# DISTRIBUTED_ARGS="
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
# "

# train-iters is originally 500000

GPT_ARGS="
    --num-layers 32 \
    --hidden-size 3072 \
    --num-attention-heads 32 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --lr 0.00015 \
    --train-iters 300 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --use-mcore-models \
    --transformer-impl local \
    --overlap-grad-reduce \
"
    # --use-mcore-models \
    # --transformer-impl local \
    # --attention-softmax-in-fp32 \
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
    --eval-iters 1
"

HANS_ARGS="
    --do-zeroing \
    --zeroing-rate $ZERO_RATE
"

python /home/lustre/NLP/Megatron-LM_clean/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH
    # --overlap-grad-reduce
