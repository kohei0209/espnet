#!/usr/bin/env bash

set -e
set -u
set -o pipefail

min_or_max=min
sample_rate=8k

. utils/parse_options.sh
. ./path.sh

if [ $# -le 1 ]; then
  echo "Arguments should be WSJ0-2MIX directory, the mixing script path and the WSJ0 path, see ../run.sh for example."
  exit 1;
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

find_transcripts=`pwd`/local/find_transcripts.pl
normalize_transcript=`pwd`/local/normalize_transcript.pl

wavdir=$1
wsj_full_wav=$2

tr="tr_${min_or_max}_${sample_rate}"
cv="cv_${min_or_max}_${sample_rate}"
tt="tt_${min_or_max}_${sample_rate}"

data=./data


for dset in $tr $cv $tt; do
    # scp files
    python local/prepare_scp_files.py \
        --simulation_metadata_dir "${data}/${dset}" \
        --scp_output_dir "${data}/${dset}" \
        --num_spk 5
    # spk2utt
    local/utt2spk_to_spk2utt.pl ${data}/${dset}/utt2spk > ${data}/${dset}/spk2utt
    # utt2category
    python local/prepare_utt2category.py \
        ${data}/${dset}/wav.scp \
        --output_dir ${data}/${dset} || exit 1;
done

# transcriptions (only for 'max' version)
if [[ "$min_or_max" = "min" ]]; then
  exit 0
fi


mkdir -p tmp
cd tmp
for i in si_tr_s si_et_05 si_dt_05; do
    cp ${wsj_full_wav}/${i}.scp .
done

# Finding the transcript files:
for x in `ls ${wsj_full_wav}/links/`; do find -L ${wsj_full_wav}/links/$x -iname '*.dot'; done > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
for f in si_tr_s si_et_05 si_dt_05; do
  cat ${f}.scp | awk '{print $1}' | ${find_transcripts} dot_files.flist > ${f}.trans1

  # Do some basic normalization steps.  At this point we don't remove OOVs--
  # that will be done inside the training scripts, as we'd like to make the
  # data-preparation stage independent of the specific lexicon used.
  noiseword="<NOISE>"
  cat ${f}.trans1 | ${normalize_transcript} ${noiseword} | sort > ${f}.txt || exit 1;
done

# change to the original path
cd ..

# preapre transcriptions for all of 2-5mix
for stage in $cv $tt; do
  python ${PWD}/local/prepare_transcription.py \
    --transcription_folder ./tmp --wavscp_folder ./data/${stage} --num_spk 5
done

rm -r tmp
