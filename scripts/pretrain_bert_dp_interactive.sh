#!/bin/bash

MEGATRON_HOME="/home/lustre/NLP/Megatron-LM_clean"
DATASET_HOME="/home/lustre/datasets/"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=0 

GPUS_PER_NODE=8
# Change for multinode config
NNODES=1
NODE_RANK=${SLURM_NODEID}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=${MEGATRON_HOME}/checkpoints/pretrain_bertl_dp/baseline/
VOCAB_FILE=${DATASET_HOME}/vocabs/bert-large-cased-vocab.txt
DATA_PATH=${DATASET_HOME}/Wikipedia-BERT/my-bert_text_sentence

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# BERT_ARGS="
#     --tensor-model-parallel-size 1 \
#     --pipeline-model-parallel-size 1 \
#     --num-layers 24 \
#     --hidden-size 1024 \
#     --num-attention-heads 16 \
#     --seq-length 512 \
#     --max-position-embeddings 512 \
#     --micro-batch-size 4 \
#     --global-batch-size 32 \
#     --lr 0.0001 \
#     --train-iters 1000000 \
#     --lr-decay-iters 990000 \
#     --lr-decay-style linear \
#     --min-lr 1.0e-5 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --fp16 \
#     --use-mcore-models \
#     --transformer-impl local \
#     --spec local \
#     --overlap-grad-reduce \
# "

BERT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --lr 1e-4 \
    --train-iters 50000 \
    --lr-decay-iters 990000 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --fp16 \
    --use-mcore-models \
    --transformer-impl local \
    --spec local \
    --overlap-grad-reduce \
"

    # --lr-warmup-fraction .01 \

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS /home/lustre/NLP/Megatron-LM_clean/pretrain_bert.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    # --load $CHECKPOINT_PATH
