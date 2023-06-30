#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh

data_root="${PWD}/data"
sciript_path="${PWD}/data/wsj0_mix/scripts/simulation_scripts/make_data.py"
_nj=36

. utils/parse_options.sh

# first make the simulation metadata
# python local/prepare_simulation_metadata.py \
#     --wav_scp_folder ${data_root} \
#     --wham_folder ../../librimix/tse1/data/wham_noise -sr 8000 \
#     --wav_output_folder ${data_root}/wsj0_mix \
#     --simulation_config_path "${PWD}/local/simulation_config.json"

# then do simulation
# for cond in tr_min_8k cv_min_8k tt_min_8k; do
#     for spk in $(seq 5); do
#         python ${sciript_path} \
#             --metadata_path "${data_root}/${cond}/simulation_metadata_${spk}spk.json" \
#             -nj ${_nj}
#         # mv ${data_root}/${cond}/spk${spk}.scp ${data_root}/${cond}/spk${spk}_backup.scp
#     done
# done

# useful for test, no need to use this
# for cond in cv_min_8k; do
#     for spk in $(seq 2 1 2); do
#         python ${sciript_path} \
#             --metadata_path "${data_root}/${cond}/simulation_metadata_${spk}spk.json" \
#             -nj 1
#     done
# done

# get noise SNR against the mixture and the actual RT60
for cond in tr_min_8k cv_min_8k tt_min_8k; do
    for spk in $(seq 5); do
        python ${PWD}/local/get_snr_and_rt60.py \
            --metadata_path "${data_root}/${cond}/simulation_metadata_${spk}spk.json" \
            --decay_db 20
    done
done

# finally prepare the scp files
# for cond in tr_min_8k cv_min_8k tt_min_8k; do
#     python ${PWD}/local/prepare_scp_files.py \
#             --simulation_metadata_dir "${data_root}/${cond}" \
#             --scp_output_dir "${data_root}/${cond}" \
#             --num_spk 5
#     ${PWD}/utt2spk_to_spk2utt.pl ${data_root}/${cond}/utt2spk > ${data_root}/${cond}/spk2utt
# done

