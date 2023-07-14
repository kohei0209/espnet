#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

data_root="${PWD}/data"

. utils/parse_options.sh


# wget --continue -O voxceleb_resnet34_LM.onnx https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx
# python -m pip install onnxruntime

for x in cv; do
    for min_or_max in min; do
            if [[ "${x}" == "tr" ]]; then
                python ./local/prepare_spk_embs_scp.py \
                    ${data_root}/${x}_${min_or_max}_8k/spk2enroll.json \
                    --onnx_path voxceleb_resnet34_LM.onnx \
                    --outdir ${data_root}/${x}_${min_or_max}_8k/spk_embs
                mv data/${x}_${min_or_max}_8k/spk_embs/spk2emb.json data/${x}_${min_or_max}_8k/spk2emb.json
            else
                for spk in $(seq 5); do
                    python ./local/prepare_spk_embs_scp.py \
                        ${data_root}/${x}_${min_or_max}_8k/backup/enroll_spk${spk}.scp \
                        --onnx_path voxceleb_resnet34_LM.onnx \
                        --outdir ${data_root}/${x}_${min_or_max}_8k/spk_embs
                    mv data/${x}_${min_or_max}_8k/enroll_spk${spk}.scp data/${x}_${min_or_max}_8k/enroll_spk${spk}.bak.scp
                    mv data/${x}_${min_or_max}_8k/spk_embs/embs.scp data/${x}_${min_or_max}_8k/enroll_spk${spk}.scp
                done
            fi
    done
done