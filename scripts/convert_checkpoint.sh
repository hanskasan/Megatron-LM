MEGATRON_HOME="/home/lustre/NLP/Megatron-LM_clean"
DATASET_HOME="/home/lustre/datasets"

CHECKPOINT_PATH=/lustre/fsw/portfolios/adlr/projects/adlr_psx_fp8/checkpoints/gpt3/fp16.gpt3.8.3b/
VOCAB_FILE=${DATASET_HOME}/vocabs/gpt2-vocab.json

python ${MEGATRON_HOME}/tools/checkpoint/convert.py \
    --model-type GPT \
    --load-dir ${CHECKPOINT_PATH} \
    --loader megatron \
    --saver megatron \
    --save-dir ${MEGATRON_HOME}/checkpoints/pp4/ \
    --vocab-file ${VOCAB_FILE} \
    --loader-transformer-impl local \
    --saver-transformer-impl local \
    --megatron-path ${MEGATRON_HOME} \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4