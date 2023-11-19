# WSJ0-Nmix recipe

Speech separation handling variable number of speakers using noisy reverberant wsj0-Nmix dataset.
Note that this this recipe assumes that data preparation of anechoic version is already finished in ```../enh_tse1```


## ESPNet installation
Please refer to this document:
https://espnet.github.io/espnet/installation.html


## Install other dependencies
We use Pyroomacoustics for simulation
```
# install simulation toolkit
pip install pyroomacoustics==0.7.3
```

## Data preparation
1. Change directory: ```cd ./egs2/wsj0_Nmix/enh_tse1```.
1. Fill your wsj0 and wsj1 paths in ```./db.sh```.
1. If you have not made the anechoic version of WSJ0-Nmix yet, please make anehoic version first in ```../enh_tse1```.
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
# we recommend to use pre-trained model if you have already trained the model with anechoic version
# please change the path of the pre-trained model parameter in the config
./run.sh --stage 6 --stop_stage 6 --enh_config enh_dptnet_eda4thlayer_1-5mix_1bce_usePretrained.yaml

# if you'd like to start training from scratch, run the following command
./run.sh --stage 6 --stop_stage 6 --enh_config enh_dptnet_eda4thlayer_1-5mix_1bce_fromscratch.yaml

# TSE training (please change init_params in config according to the exp directory name you set)
./run.sh --stage 6 --stop_stage 6 --enh_config ./conf/tuning/tse_dptnet_eda4thlayer_adapt5thlayer_usePretrained.yaml
```

## Inference of separation and TSE
```
# Run both separation and TSE inference
./run.sh --stage 7 --stop_stage 8 --enh_config ./conf/tuning/tse_dptnet_eda4thlayer_adapt5thlayer_usePretrained.yaml --inf_nums "1 2 3 4 5" --inference_model valid.loss.ave_5best.pth
```

## ASR evaluation using Whisper Large V2 (for "max" version)
Note that this stage does not work when "min" version is used
```
# Run both separation and TSE inference
./run.sh --stage 9 --stop_stage 9 --enh_config ./conf/tuning/tse_dptnet_eda4thlayer_adapt5thlayer_usePretrained.yaml --inf_nums "1 2 3 4 5"
```