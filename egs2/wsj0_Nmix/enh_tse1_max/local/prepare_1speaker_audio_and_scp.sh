#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh

data_folder=${PWD}/data
wsj0_metadata_folder=${data_folder}/wsj0_mix/scripts/metadata2
wav_output_folder=${data_folder}/wsj0_mix/1speakers

min_or_max=min
sample_rate=8k

. utils/parse_options.sh

sample_rate_int=${sample_rate%"k"}
sample_rate_int=$((sample_rate_int * 1000))
python ${PWD}/local/prepare_1speaker_audio_and_scp.py \
    --wsj0_metadata_folder ${wsj0_metadata_folder} \
    --wav_output_folder ${wav_output_folder}  --scp_output_folder ${data_folder} -sr ${sample_rate_int}

for cond in tr cv tt; do
    mkdir -p ${data_folder}/${cond}_${min_or_max}_${sample_rate}/include_1mix
    # prepare enroll utterances
    cp ${data_folder}/${cond}_${min_or_max}_${sample_rate}/spk2enroll.json ${data_folder}/${cond}_${min_or_max}_${sample_rate}/include_1mix
    if [ "$cond" = "tr" ]; then
        train=true
    else
        train=false
    fi
    python local/prepare_wsj_enroll.py \
        ${data_folder}/${cond}_${min_or_max}_${sample_rate}/tmp_1mix.scp \
        ${data_folder}/${cond}_${min_or_max}_${sample_rate}/include_1mix/spk2enroll.json \
        --num_spk 5 --train ${train} --seed 1 \
        --output_dir ${data_folder}/${cond}_${min_or_max}_${sample_rate}/include_1mix || exit 1;
    for spk in $(seq 5); do
        cat ${data_folder}/${cond}_${min_or_max}_${sample_rate}/enroll_spk${spk}.scp >> ${data_folder}/${cond}_${min_or_max}_${sample_rate}/include_1mix/enroll_spk${spk}.scp
    done

    # make scp files
    sort -k1 ${data_folder}/${cond}_${min_or_max}_${sample_rate}/tmp_1mix.scp > ${data_folder}/${cond}_${min_or_max}_${sample_rate}/include_1mix/wav.scp
    cat ${data_folder}/${cond}_${min_or_max}_${sample_rate}/wav.scp >> ${data_folder}/${cond}_${min_or_max}_${sample_rate}/include_1mix/wav.scp
    rm ${data_folder}/${cond}_${min_or_max}_${sample_rate}/tmp_1mix.scp
    for spk in $(seq 5); do
        sort -k1 ${data_folder}/${cond}_${min_or_max}_${sample_rate}/tmp_1mix_spk${spk}.scp > ${data_folder}/${cond}_${min_or_max}_${sample_rate}/include_1mix/spk${spk}.scp
        cat ${data_folder}/${cond}_${min_or_max}_${sample_rate}/spk${spk}.scp >> ${data_folder}/${cond}_${min_or_max}_${sample_rate}/include_1mix/spk${spk}.scp
        rm ${data_folder}/${cond}_${min_or_max}_${sample_rate}/tmp_1mix_spk${spk}.scp
    done

    # utt2category
    python local/prepare_utt2category.py \
        ${data_folder}/${cond}_${min_or_max}_${sample_rate}/include_1mix/wav.scp \
        --output_dir ${data_folder}/${cond}_${min_or_max}_${sample_rate}/include_1mix || exit 1;
done
