# Data preparation of wsj0-Nmix dataset

Data preparation of wsj0-Nmix dataset: https://github.com/mpariente/pywsj0-mix.
Currently we do not have a recipe and only support the data preparation


## ESPNet installation
Please refer to this document:
https://espnet.github.io/espnet/installation.html


## Data preparation
1. Change directory: ```cd ./egs2/wsj0_Nmix/enh_tse1```.
1. Fill your wsj0 and wsj1 paths in ```./db.sh```.
1. Change ```min_or_max``` and ```sample_rate``` parameters in ```./run.sh``` according to your purpose.
1. Run the data preparation code as follows.
```
# data is generated under ./data/wsj0_mix/?speakers
./run.sh --stage 1 --stop_stage 1
```
