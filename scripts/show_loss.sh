readonly REPORT_NAME=oscillate_0_75_15Kwarm_overlap_protect37
readonly READ_FROM=/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/reports/pretrain_gpt1.7B_dp/${REPORT_NAME}.report
readonly WRITE_TO=/lustre/fsw/portfolios/nvr/users/hkasan/workspace/NLP/Megatron-LM_clean/greped/pretrain_gpt1.7B_dp/${REPORT_NAME}.report

grep 'lm loss' ${READ_FROM} | grep 'number of skipped iterations' | awk '{print $29}' > ${WRITE_TO}