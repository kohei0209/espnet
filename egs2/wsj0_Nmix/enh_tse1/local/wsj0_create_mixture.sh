#!/usr/bin/env bash

# Copyright  2018  Johns Hopkins University (Author: Xuankai Chang)
#            2020  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0

. utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "Usage: $0 <dir> <wsj0-path> <wsj0-full-wav> <wsj0-2mix-wav>"
  echo " where <dir> is download space,"
  echo " <wsj0-path> is the original wsj0 path"
  echo " <wsj0-full-wav> is wsj0 full wave files path, <wsj0-2mix-wav> is wav generation space."
  echo "Note: this script won't actually re-download things if called twice,"
  echo "because we use the --continue flag to 'wget'."
  echo "Note: this script can be used to create wsj0_2mix and wsj_2mix corpus"
  echo "Note: <wsj0-full-wav> contains all the wsj0 (or wsj) utterances in wav format,"
  echo "and the directory is organized according to"
  echo "  scripts/mix_2_spk_tr.txt, scripts/mix_2_spk_cv.txt and mix_2_spk_tt.txt"
  echo ", which are the mixture combination schemes."
  exit 1;
fi

dir=$1
wsj0_path=$2
wsj_full_wav=$3
wsj_mix_wav=$4
nsrc=$5


# if ! which matlab >/dev/null 2>&1; then
#     echo "matlab not found."
#     exit 1
# fi

echo "Downloading WSJ0_mixture scripts."
mkdir -p ${dir}
mkdir -p ${wdir}/log

# git clone https://github.com/mpariente/pywsj0-mix.git ${dir}


echo "WSJ0 wav file."
# local/convert2wav.sh ${wsj0_path} ${wsj_full_wav} || exit 1;

echo "Creating Mixtures."

python_cmd="python generate_wsjmix.py -p ${wsj_full_wav} -o ${wsj_mix_wav} -n ${nsrc} -sr 8000 --len_mode min"

mixfile=${dir}/mix_python.sh
echo "#!/usr/bin/env bash" > $mixfile
echo "cd ${dir}" >> $mixfile
echo $python_cmd >> $mixfile
chmod +x $mixfile

# Run python
# (This may take ~6 hours to generate both min and max versions
#  on Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz)
echo "Log is in ${dir}/mix.log"
$train_cmd ${dir}/mix.log $mixfile
