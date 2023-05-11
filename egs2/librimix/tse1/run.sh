#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=8k # If using 8k, please make sure `spk2enroll.json` points to 8k audios as well
min_or_max=min  # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.

train_set="train"
valid_set="dev"
test_sets="test "

# config=./conf/tuning/train_enh_tse_asenet.yaml
# config=./conf/tuning/train_enh_tse_td_speakerbeam.yaml
# config=./conf/tuning/train_enh_tse_td_speakerbeam_with_attention.yaml
# config=./conf/tuning/train_enh_tse_td_speakerbeam_simpleattn.yaml
config=./conf/tuning/train_enh_tse_td_speakerbeam_simpleattn.yaml

./enh.sh \
    --is_tse_task false \
    --is_tse_and_ss_task true \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --ref_num 2 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --lang en \
    --ngpu 1 \
    --enh_config "${config}" \
    "$@"

# ./enh.sh \
#     --is_tse_and_ss_task true \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_sets}" \
#     --fs "${sample_rate}" \
#     --ref_num 2 \
#     --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
#     --lang en \
#     --ngpu 1 \
#     --enh_config "${config}" \
#     "$@"
