#!/usr/bin/env bash

exp=$1 # e.g., "exp/enh_train_enh_bsrnn_medium_noncausal_raw". replace this with your exp directory

split="validation"
eval_noisy_data=false

# inference (for both validation and test sets)
if [[ "$eval_noisy_data" = false && ! -e "${exp}/enhanced_${split}" ]]; then
    ./run.sh --stage 7 --stop-stage 7 --enh_exp ${exp} --gpu_inference true --inference_model valid.loss.best.pth --inference_nj 1
elif [[ "$eval_noisy_data" = false && -e "${exp}/enhanced_${split}" ]]; then
    echo "${exp}/${split} exists. Remove the directory if you want to run inference again."
fi

# scoring (only for the validation set)
. ./path2.sh

if [ "$eval_noisy_data" = true ]; then
    inf_scp="dump/raw/${split}/wav.scp"
else
    inf_scp="${exp}/enhanced_${split}/spk1.scp"
fi
output_dir=$(dirname "$inf_scp")

# non-intrusive metric (DNSMOS)
python urgent2025_challenge/evaluation_metrics/calculate_nonintrusive_dnsmos.py \
    --inf_scp ${inf_scp} \
    --output_dir ${output_dir}/scoring_dnsmos \
    --device cuda \
    --convert_to_torch True \
    --primary_model urgent2025_challenge/DNSMOS/DNSMOS/sig_bak_ovr.onnx \
    --p808_model urgent2025_challenge/DNSMOS/DNSMOS/model_v8.onnx

# non-intrusive metric (NISQA)
python urgent2025_challenge/evaluation_metrics/calculate_nonintrusive_nisqa.py \
    --inf_scp ${inf_scp} \
    --output_dir ${output_dir}/scoring_nisqa \
    --device cuda \
    --nisqa_model urgent2025_challenge/lib/NISQA/weights/nisqa.tar

# non-intrusive metric (UTMOS)
python urgent2025_challenge/evaluation_metrics/calculate_nonintrusive_mos.py \
    --inf_scp ${inf_scp} \
    --output_dir ${output_dir}/scoring_utmos \
    --device cuda \
    --utmos_tag utmos22_strong

# intrusive SE metrics (calculated on CPU)
python urgent2025_challenge/evaluation_metrics/calculate_intrusive_se_metrics.py \
    --ref_scp dump/raw/${split}/spk1.scp \
    --inf_scp ${inf_scp} \
    --output_dir ${output_dir}/scoring \
    --nj 8 \
    --chunksize 500

# downstream-task-independent metric (SpeechBERTScore)
python urgent2025_challenge/evaluation_metrics/calculate_speechbert_score.py \
    --ref_scp dump/raw/${split}/spk1.scp \
    --inf_scp ${inf_scp} \
    --output_dir ${output_dir}/scoring_speech_bert_score \
    --device cuda

# downstream-task-independent metric (LPS)
python urgent2025_challenge/evaluation_metrics/calculate_phoneme_similarity.py \
    --ref_scp dump/raw/${split}/spk1.scp \
    --inf_scp ${inf_scp} \
    --output_dir ${output_dir}/scoring_phoneme_similarity \
    --device cuda

# downstream-task-dependent metric (SpkSim)
python urgent2025_challenge/evaluation_metrics/calculate_speaker_similarity.py \
    --ref_scp dump/raw/${split}/spk1.scp \
    --inf_scp ${inf_scp} \
    --output_dir ${output_dir}/scoring_speaker_similarity \
    --device cuda

# downstream-task-dependent metric (WER or 1-WAcc)
python urgent2025_challenge/evaluation_metrics/calculate_wer.py \
    --meta_tsv dump/raw/${split}/text \
    --inf_scp ${inf_scp} \
    --utt2lang dump/raw/${split}/utt2lang \
    --output_dir ${output_dir}/scoring_wer \
    --device cuda