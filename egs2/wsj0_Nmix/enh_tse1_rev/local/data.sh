#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--min_or_max <min/max>] [--sample_rate <8k/16k>]
  optional argument:
    [--min_or_max]: min (Default), max
    [--sample_rate]: 8k (Default), 16k
EOF
)

. ./db.sh

wsj_full_wav=$PWD/data/wsj0
wsj_mix_wav=$PWD/data/wsj0_mix
wsj_mix_scripts=$PWD/data/wsj0_mix/scripts
wsj_scp_output_dir=$PWD/data
anechoic_mix_dir="$(dirname "${PWD}")/enh_tse1"  # /path/to/enh_tse1, like  ${PWD}/../enh_tse1
wham_noise=


other_text=data/local/other_text/text
nlsyms=data/nlsyms.txt
min_or_max=min
sample_rate=8k

stage=0
stop_stage=100

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' of db.sh"
    exit 1
fi
if [ ! -e "${WSJ1}" ]; then
    log "Fill the value of 'WSJ1' of db.sh"
    exit 1
fi
if [ -z "${anechoic_mix_dir}" ]; then
    log "Please fill the path of anechoic_mix_dir"
fi
if [ ! -e "${anechoic_mix_dir}" ]; then
    log "Please make anechoic data first by running recipe in enh_tse1"
    exit 1
fi

train_set="tr_"${min_or_max}_${sample_rate}
train_dev="cv_"${min_or_max}_${sample_rate}
recog_set="tt_"${min_or_max}_${sample_rate}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    dir="./data"
    mkdir -p ${dir}
    log "stage 0: WHAM noise download and anechoic make 1-mix data"
    if [ -z "$wham_noise" ]; then
        if [ $(ls ${dir}/wham_noise 2>/dev/null | wc -l) -eq 4 ]; then
            echo "'${dir}/wham_noise/' already exists. Skipping..."
        else
            # 17.65 GB unzipping to 35 GB
            wham_noise_url=https://storage.googleapis.com/whisper-public/wham_noise.zip
            wget --continue -O $dir/wham_noise.zip ${wham_noise_url}
            unzip ${dir}/wham_noise.zip -d ${dir}
        fi
        wham_noise=${dir}/wham_noise
    # else
    #     # make symbolic link
    #     ln -s "$wham_noise" ${PWD}/data
    fi

    # make anechoic 1-mix
    if [ ! -e "${anechoic_mix_dir}/data/wsj0_mix/1speakers" ]; then
        sample_rate_int=$((${sample_rate%"k"} * 1000))
        python local/prepare_1mix.py \
            --wsj0_metadata_folder ${anechoic_mix_dir}/data/wsj0_mix/scripts/metadata2 \
            --wav_output_folder ${anechoic_mix_dir}/data/wsj0_mix/1speakers  \
            -sr ${sample_rate_int}
    else
        log "${anechoic_mix_dir}/data/wsj0_mix/1speakers already exists. Skipping..."
    fi

    # make symbolic link to anechoic mixture
    ln -s ${anechoic_mix_dir}/data/wsj0_mix ./data/anechoic_wsj0_mix
    ln -s ${anechoic_mix_dir}/data/wsj0 ${wsj_full_wav}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data simulation"
    ### This part is for WSJ0 mix
    for task in tr cv tt; do
        # simulation
        local/simulation.sh \
            --min-or-max ${min_or_max} --sample-rate ${sample_rate} --task ${task} || exit 1;
    done
    # other data prep
    local/wsj0mix_data_prep.sh \
        --min-or-max ${min_or_max} --sample-rate ${sample_rate} \
        ${wsj_mix_wav}/wav${sample_rate}/${min_or_max} ${wsj_full_wav} || exit 1;
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Prepare language information"
    ### Also need wsj corpus to prepare language information
    ### This is from Kaldi WSJ recipe
    log "local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?"
    local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?
    log "local/wsj_format_data.sh"
    local/wsj_format_data.sh
    log "mkdir -p data/wsj"
    mkdir -p data/wsj
    log "mv data/{dev_dt_*,local,test_dev*,test_eval*,train_si284} data/wsj"
    mv data/{dev_dt_*,local,test_dev*,test_eval*,train_si284} data/wsj

    log "Prepare text from lng_modl dir: ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z -> ${other_text}"
    mkdir -p "$(dirname ${other_text})"

    # NOTE(kamo): Give utterance id to each texts.
    zcat ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | \
        grep -v "<" | tr "[:lower:]" "[:upper:]" | \
        awk '{ printf("wsj1_lng_%07d %s\n",NR,$0) } ' > ${other_text}



    log "Create non linguistic symbols: ${nlsyms}"
    cut -f 2- data/wsj/train_si284/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Prepare WSJMix enroll utterances"

    for task in tr cv tt;
    do
        # copy scp files from local while fixing some infomation
        # e.g., min->max, 8k->16k and make them absolute paths
        python local/copy_enroll_files.py \
            --source_folder ./local/enroll_scp/${task}_min_8k \
            --target_folder ${wsj_scp_output_dir}/${task}_${min_or_max}_${sample_rate} \
            --source_words ../enh_tse1/ /min/ /wav8k/ \
            --target_words "$(dirname "${PWD}")/enh_tse1/" /${min_or_max}/ /wav${sample_rate}/
            # --target_words ${PWD}/ /${min_or_max}/ /wav${sample_rate}/

        # for training set, we prepare scp files here
        if [ "$task" = "tr" ]; then
            python local/prepare_wsj_enroll.py \
                ${wsj_scp_output_dir}/${task}_${min_or_max}_${sample_rate}/wav.scp \
                ${wsj_scp_output_dir}/${task}_${min_or_max}_${sample_rate}/spk2enroll.json \
                --num_spk 5 --train true \
                --output_dir ${wsj_scp_output_dir}/${task}_${min_or_max}_${sample_rate} || exit 1;
        fi

    done
fi



