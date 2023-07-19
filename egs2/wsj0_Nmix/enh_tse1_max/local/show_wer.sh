#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

exp_dir=$1
task=$2
inf_nums=$3

for nmix in $inf_nums; do
    for suffix in with without; do
        echo "${suffix}_true_nspk  ${nmix}mix data"
        python ./local/merge_wer.py  "${exp_dir}/enhanced_tt_max_8k/${task}/${nmix}mix/${suffix}_true_numspk/whisper" ${nmix}
    done
done
