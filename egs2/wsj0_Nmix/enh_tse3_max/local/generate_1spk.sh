#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh

wsj0_metadata_folder=./data/wsj0_mix/scripts/metadata2
wav_output_folder=/mnt/kiso-qnap/saijo/universal_se/espnet_use/egs2/wsj0_Nmix/enh_tse1/data/wsj0_mix/1speakers
scp_output_folder=./data/data_of_enh1
sample_rate=8000

. utils/parse_options.sh

# python data/wsj0_mix/scripts/generate_1spk_scp.py \
#     --wsj0_metadata_folder ${wsj0_metadata_folder} \
#     --wav_output_folder ${wav_output_folder}  --scp_output_folder ${scp_output_folder} -sr ${sample_rate}

# needs to first copy enh_tse1/data/${cond}

for cond in tr_min_8k cv_min_8k tt_min_8k; do
    mkdir -p ${PWD}/data/${cond}
    sort -k1 ${scp_output_folder}/${cond}/tmp_1mix.scp > ${scp_output_folder}/${cond}/tmp_1mix_2.scp
    cat ${scp_output_folder}/${cond}/tmp_1mix_2.scp > ${PWD}/data/${cond}/wav.scp
    cat ${scp_output_folder}/${cond}/wav.scp >> ${PWD}/data/${cond}/wav.scp
    for spk in $(seq 5); do
        sort -k1 ${scp_output_folder}/${cond}/tmp_1mix_spk${spk}.scp > ${scp_output_folder}/${cond}/tmp_1mix_spk${spk}_2.scp
        cat ${scp_output_folder}/${cond}/tmp_1mix_spk${spk}_2.scp > ${PWD}/data/${cond}/spk${spk}.scp
        cat ${scp_output_folder}/${cond}/spk${spk}.scp >> ${PWD}/data/${cond}/spk${spk}.scp
    done
done