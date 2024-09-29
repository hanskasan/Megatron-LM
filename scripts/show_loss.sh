# readonly MODEL_NAME=pretrain_bertl_ablation
# readonly MODEL_NAME=pretrain_bertl_dp
# readonly MODEL_NAME=pretrain_bertl_dp/ideal-random/
# readonly MODEL_NAME=pretrain_bertl_fp32_ablation
# readonly MODEL_NAME=pretrain_bertl_reliability
# readonly MODEL_NAME=pretrain_gpt1.7B_dp
# readonly MODEL_NAME=pretrain_gpt-medium_dp
readonly MODEL_NAME=pretrain_gpt-medium_dp/ideal-random

# readonly REPORT_NAME=baseline_fp32
# readonly REPORT_NAME=baseline_fp32_overlap_nodrop
# readonly REPORT_NAME=top25_fp32_overlap_nodrop
# readonly REPORT_NAME=top90_fp32_nodrop
# readonly REPORT_NAME=ideal-random50_fp32
readonly REPORT_NAME=ideal-random50_protect-pos-embedding_layer0_fc2_linproj_finalln_10Kwarm_fp32
# readonly REPORT_NAME=ideal-random50_protectsmall-poolerweight-denseweight-linprojweights_layer0
# readonly REPORT_NAME=ideal-random50_protectsmall-poolerweight-denseweight-linprojweights_layer0
# readonly REPORT_NAME=skip25_layer0to2_smartbucket
# readonly REPORT_NAME=skip25_layer0to2_protectsmall
# readonly REPORT_NAME=test50

# readonly REPORT_NAME=ideal-random50_protectsmall_linproj_fc2-weight_pos-embedding_fp32

readonly READ_FROM=/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/reports/${MODEL_NAME}/${REPORT_NAME}.report

WRITE_TO=/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/greped/${MODEL_NAME}/${REPORT_NAME}.report
grep 'lm loss' ${READ_FROM} | grep 'number of skipped iterations' | awk '{print $29}' > ${WRITE_TO}

WRITE_TO=/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/greped/${MODEL_NAME}/validation_${REPORT_NAME}.report
grep 'validation loss' ${READ_FROM} | grep -v "set" | awk '{print $10}' > ${WRITE_TO}

WRITE_TO=/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/greped/${MODEL_NAME}/lr_${REPORT_NAME}.report
grep 'learning rate:' ${READ_FROM} | awk '{print $20}' > ${WRITE_TO}