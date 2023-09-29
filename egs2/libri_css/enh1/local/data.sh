#!/usr/bin/env bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Chenda Li, Wangyou Zhang)
# Apache 2.0
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

# assuming user already has libricss dataset
libricss_root=/mnt/kiso-qnap/saijo/libri_css
sample_rate=16k

stage=0
stop_stage=100

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

# prepare wav.scp file
mkdir -p "${PWD}/data"
python local/prepare_libricss_wavscp.py \
    --libricss_root ${libricss_root} --scp_output_dir "${PWD}/data"
sort -k1 "${PWD}/data/wav.scp.bak" > "${PWD}/data/wav.scp"
rm "${PWD}/data/wav.scp.bak"

# TODO: prepare utt2spk and enrollment utterances

# prepare text file
sort -k1 "${libricss_root}/exp/data/monaural/utterances/utterance_transcription.txt" > "${PWD}/data/utterance_transcription.txt"

# make dump folder
ln -s "${PWD}/data" "${PWD}/dump"
