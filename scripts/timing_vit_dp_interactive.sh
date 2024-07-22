#!/bin/bash

MEGATRON_HOME="/home/lustre/NLP/Megatron-LM/"
DATASET_HOME="/home/lustre/datasets/"

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Training and validation paths should each point to a folder where each
# sub-folder contains a collection of images in jpg or png format
# e.g. If using imagenet, one train image might be, train_data/n01688243/n01688243_11301.JPEG
DATA_PATH_TRAIN=${DATASET_HOME}/imagenet_1k/train/
DATA_PATH_VAL=${DATASET_HOME}/imagenet_1k/val/
CHECKPOINT_PATH=${MEGATRON_HOME}/checkpoints/vit/

# Change for multinode config
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=1995
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Number of layers is originally 56 (ViT-e)

CLASSIFIER_ARGS="
   	--tensor-model-parallel-size 1 \
    --num-layers 48 \
    --hidden-size 1792 \
    --num-attention-heads 16 \
    --patch-dim 4 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --img-h 224 \
    --img-w 224 \
    --mask-factor 1.0 \
    --fp16 \
    --train-iters 300 \
    --lr-decay-style cosine \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --lr 0.0005 \
    --min-lr 0.00001 \
    --attention-dropout 0.0 \
    --weight-decay 0.05 \
    --lr-warmup-iters 100 \
    --clip-grad 1.0 \
    --no-gradient-accumulation-fusion \
    --use-mcore-models \
    --transformer-impl local
"
    # --attention-softmax-in-fp32 \

    # --use-mcore-models \
    # --transformer-impl local
    
    # --recompute-granularity full \
    # --recompute-method uniform \
    # --recompute-num-layers 4

DATA_ARGS="
    --tokenizer-type NullTokenizer \
    --vocab-size 0 \
    --data-path $DATA_PATH_TRAIN $DATA_PATH_VAL \
    --no-data-sharding \
    --split 949,50,1 \
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 1
"

torchrun $DISTRIBUTED_ARGS /home/lustre/NLP/Megatron-LM_clean/pretrain_vision_classify.py \
    $CLASSIFIER_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    # --save $CHECKPOINT_PATH
    # $HANS_ARGS \
    # --load $CHECKPOINT_PATH
    # --overlap-grad-reduce