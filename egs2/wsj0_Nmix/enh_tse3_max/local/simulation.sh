#!/usr/bin/env bash

## NOTE ##
# Simulation needs anechoic version of WSJ0-Mix
# First do ./run.sh --stage 1 --stop_stage 1 in enh_tse1

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh

anechoic_data_folder="../enh_tse1_max/data"
data_root="${PWD}/data"
wham_folder="../../librimix/tse1/data/wham_noise"

sample_rate=8k
min_or_max=max

_nj=36

. utils/parse_options.sh

sample_rate_int=${sample_rate%"k"}
sample_rate_int=$((sample_rate_int * 1000))

tr="tr_${min_or_max}_${sample_rate}"
cv="cv_${min_or_max}_${sample_rate}"
tt="tt_${min_or_max}_${sample_rate}"

# for cond in $tr $cv $tt; do
#     echo "copy scp files from ${anechoic_data_folder}"
#     mkdir -p ./data/${cond}
#     cp ${anechoic_data_folder}/${cond}/include_1mix/* ./data/${cond}/
# done

for cond in tr cv; do
    python local/copy_simulation_metadata.py \
        --source_folder ../enh_tse3/data/${cond}_min_${sample_rate}/ \
        --target_folder ./data/${cond}_${min_or_max}_${sample_rate}/ \
        --source_words /enh_tse3/ /enh_tse1/ /min/ \
        --target_words /enh_tse3_max/ /enh_tse1_max/ /max/     \
        --anechoic_folder ${anechoic_data_folder}/${cond}_${min_or_max}_${sample_rate}/include_1mix/ --nmix 5
done

# then do simulation
for cond in $tr $cv $tt; do
    for spk in $(seq 5); do
        python local/make_data.py \
            --metadata_path "${data_root}/${cond}/simulation_metadata_${spk}spk.json" \
            -nj ${_nj}

        # mv ${data_root}/${cond}/spk${spk}.scp ${data_root}/${cond}/spk${spk}_backup.scp

        # get noise SNR against the mixture and the actual RT60
        python ${PWD}/local/get_snr_and_rt60.py \
            --metadata_path "${data_root}/${cond}/simulation_metadata_${spk}spk.json" \
            --decay_db 20
    done

    python ${PWD}/local/prepare_scp_files.py \
            --simulation_metadata_dir "${data_root}/${cond}_${min_or_max}_${sample_rate}" \
            --scp_output_dir "${data_root}/${cond}_${min_or_max}_${sample_rate}" \
            --num_spk 5
    ${PWD}/utt2spk_to_spk2utt.pl ${data_root}/${cond}_${min_or_max}_${sample_rate}/utt2spk > ${data_root}/${cond}_${min_or_max}_${sample_rate}/spk2utt
done

# useful for test, no need to use this
# for cond in cv_min_8k; do
#     for spk in $(seq 2 1 2); do
#         python ${sciript_path} \
#             --metadata_path "${data_root}/${cond}/simulation_metadata_${spk}spk.json" \
#             -nj 1
#     done
# done

# finally prepare the scp files
for cond in $tr $cv $tt; do
    python ${PWD}/local/prepare_scp_files.py \
            --simulation_metadata_dir "${data_root}/${cond}" \
            --scp_output_dir "${data_root}/${cond}" \
            --num_spk 5
    ${PWD}/utt2spk_to_spk2utt.pl ${data_root}/${cond}/utt2spk > ${data_root}/${cond}/spk2utt
done

