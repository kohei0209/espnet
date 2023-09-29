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

data_root="${PWD}/data"

sample_rate=8k
min_or_max=min
task=

_nj=32

. utils/parse_options.sh

sample_rate_int=${sample_rate%"k"}
sample_rate_int=$((sample_rate_int * 1000))

mkdir -p ${data_root}/${task}_${min_or_max}_${sample_rate}

python local/copy_simulation_metadata.py \
    --source_folder ./local/simulation_metadata/${task}_${min_or_max}_8k \
    --target_folder ${data_root}/${task}_${min_or_max}_${sample_rate} \
    --source_words ./ /min/ /wav8k/ \
    --target_words ${PWD}/ /${min_or_max}/ /wav${sample_rate}/

for spk in $(seq 3 5); do
    # make data directories
    mkdir -p "${data_root}/wsj0_mix/${spk}speakers/wav${sample_rate}/${min_or_max}/${task}/mix"
    mkdir -p "${data_root}/wsj0_mix/${spk}speakers/wav${sample_rate}/${min_or_max}/${task}/noise"
    for s in $(seq ${spk}); do
        mkdir -p "${data_root}/wsj0_mix/${spk}speakers/wav${sample_rate}/${min_or_max}/${task}/s${s}"
        mkdir -p "${data_root}/wsj0_mix/${spk}speakers/wav${sample_rate}/${min_or_max}/${task}/reverb_s${s}"
        mkdir -p "${data_root}/wsj0_mix/${spk}speakers/wav${sample_rate}/${min_or_max}/${task}/rir_s${s}"
    done

    # simulate noisy reverberant mixtures
    python local/simulation.py \
        --metadata_path "${data_root}/${task}_${min_or_max}_${sample_rate}/simulation_metadata_${spk}spk.json" \
        -nj ${_nj}

    # get noise SNR against the mixture and the actual RT60
    python ${PWD}/local/get_snr_and_rt60.py \
        --metadata_path "${data_root}/${task}_${min_or_max}_${sample_rate}/simulation_metadata_${spk}spk.json" \
        --decay_db 20
done
