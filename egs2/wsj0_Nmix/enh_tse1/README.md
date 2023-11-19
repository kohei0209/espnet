# WSJ0-Nmix recipe

Speech separation handling variable number of speakers using wsj0-Nmix dataset: https://github.com/mpariente/pywsj0-mix.


## ESPNet installation
Please refer to this document:
https://espnet.github.io/espnet/installation.html


## Install other dependencies
We use Transformers for ASR evaluation
```
# install transformers
pip install transformers
```

## Data preparation
1. Change directory: ```cd ./egs2/wsj0_Nmix/enh_tse1```.
1. Fill your wsj0 and wsj1 paths in ```./db.sh```.
1. Change ```min_or_max``` and ```sample_rate``` parameters in ```./run.sh``` according to your purpose.
1. The data preparation and collecting stats is done as follows.
```
# data is generated under ./data/wsj0_mix/?speakers
./run.sh --stage 1 --stop_stage 5 --enh_args "--task tse" --audio_format wav
```

## Training
Training can be done as follows.
The default config is run with a signle RTX3090 when separation training and with a single 2080Ti when TSE training, respectively.
```
# separation training
./run.sh --stage 6 --stop_stage 6 --enh_config ./conf/tuning/enh_dptnet_eda4thlayer_2-5mix_1bce.yaml

# TSE training
./run.sh --stage 6 --stop_stage 6 --enh_config ./conf/tuning/tse_dptnet_eda4thlayer_2-5mix_1bce_sisnr_enhanced_spkselect.yaml
```

## Inference of separation and TSE
```
# Run both separation and TSE inference
./run.sh --stage 7 --stop_stage 8 --enh_config ./conf/tuning/tse_dptnet_eda4thlayer_2-5mix_1bce_sisnr_enhanced_spkselect.yaml --inf_nums "2 3 4 5" --inference_model valid.loss.ave_5best.pth
```

## ASR evaluation using Whisper Large V2 (for "max" version)
Note that this stage does not work when "min" version is used
```
# Run both separation and TSE inference
./run.sh --stage 9 --stop_stage 9 --enh_config ./conf/tuning/tse_dptnet_eda4thlayer_2-5mix_1bce_sisnr_enhanced_spkselect.yaml --inf_nums "2 3 4 5"
```