# readonly MODEL_NAME=pretrain_bertl_dp
readonly MODEL_NAME=pretrain_gpt1.7B_dp

# readonly REPORT_NAME=oscillate_0_75_25Kwarm_overlap_nodrop
readonly REPORT_NAME=test50

readonly READ_FROM=/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/reports/${MODEL_NAME}/${REPORT_NAME}.report
readonly WRITE_TO=/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/greped/${MODEL_NAME}/${REPORT_NAME}.report

grep 'lm loss' ${READ_FROM} | grep 'number of skipped iterations' | awk '{print $29}' > ${WRITE_TO}