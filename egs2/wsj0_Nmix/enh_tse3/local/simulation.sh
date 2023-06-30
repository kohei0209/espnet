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

anechoic_data_folder="../enh_tse1/data"
data_root="${PWD}/data"
wham_folder="../../librimix/tse1/data/wham_noise"

sample_rate=8k
min_or_max=min

_nj=36

. utils/parse_options.sh

sample_rate_int=${sample_rate%"k"}
sample_rate_int=$((sample_rate_int * 1000))

# first make the simulation metadata
python local/prepare_simulation_metadata.py \
    --wav_scp_folder ${anechoic_data_folder} \
    --wham_folder ${wham_folder} -sr ${sample_rate_int} \
    --wav_output_folder ${data_root}/wsj0_mix \
    --simulation_config_path "${PWD}/local/simulation_config.json"

# then do simulation
for cond in tr cv tt; do
    for spk in $(seq 5); do
        python local/make_data.py \
            --metadata_path "${data_root}/${cond}_${min_or_max}_${sample_rate}/simulation_metadata_${spk}spk.json" \
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
# for cond in tr_min_8k cv_min_8k tt_min_8k; do
#     python ${PWD}/local/prepare_scp_files.py \
#             --simulation_metadata_dir "${data_root}/${cond}" \
#             --scp_output_dir "${data_root}/${cond}" \
#             --num_spk 5
#     ${PWD}/utt2spk_to_spk2utt.pl ${data_root}/${cond}/utt2spk > ${data_root}/${cond}/spk2utt
# done

