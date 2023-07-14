from pathlib import Path
import json
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm

from espnet2.fileio.read_text import read_2columns_text

"""
This script is used to copy and change simulation configuration from min version to max version

python local/copy_simulation_metadata.py \
    --source_folder ../enh_tse3/data/tt_min_8k/ --target_folder ./data/tt_max_8k/ \
    --source_words /enh_tse3/ /enh_tse1/ /min/ --target_words /enh_tse3_max/ /enh_tse1_max/ /max/ \
    --anechoic_folder ../enh_tse1_max/data/tt_max_8k/include_1mix/ --nmix 5
"""

FS_ORIG_NOISE=16000

parser = argparse.ArgumentParser()
parser.add_argument("--source_folder", type=Path, required=True)
parser.add_argument("--target_folder", type=Path, required=True)
parser.add_argument(
    "--source_words",
    nargs="*", type=str, required=True,
    help="Specify the word that you would like to replace from"
)
parser.add_argument(
    "--target_words",
    nargs="*", type=str, required=True,
    help="Specify the word that you would like to replace with"
)
parser.add_argument("--anechoic_folder", type=Path, required=True)
parser.add_argument("--nmix", type=int, required=True)
args = parser.parse_args()


for spk in range(args.nmix):
    np.random.seed(spk)
    wav_scp = read_2columns_text(args.anechoic_folder / "wav.scp")
    # load original file to be copied
    original_file_path = args.source_folder / f"simulation_metadata_{spk + 1}spk.json"
    with open(original_file_path, "r") as f:
        original_metadata = json.load(f)
    new_metadata = {}
    noise_duration_path_list = []
    sss = 0
    for key, info in tqdm(original_metadata.items()):
        new_info = info
        for (source_word, target_word) in zip(args.source_words, args.target_words):
            for tag in new_info["audio_paths"].keys():
                if isinstance(new_info["audio_paths"][tag], list):
                    for i in range(len(new_info["audio_paths"][tag])):
                        new_info["audio_paths"][tag][i] = new_info["audio_paths"][tag][i].replace(source_word, target_word)
                else:
                    new_info["audio_paths"][tag] = new_info["audio_paths"][tag].replace(source_word, target_word)
            for i in range(len(new_info["drysrc_paths"])):
                new_info["drysrc_paths"][i] = new_info["drysrc_paths"][i].replace(source_word, target_word)

        # get mixture length
        speech_sf_info = sf.info(wav_scp[key])
        speech_duration = speech_sf_info.frames
        speech_samplerate = speech_sf_info.samplerate
        new_info["speech_duration"] = speech_duration

        # noise duration check
        noise_sf_info = sf.info(info["original_noise_path"])
        noise_duration = noise_sf_info.frames
        noise_samplerate = noise_sf_info.samplerate
        noise_duration_path_list.append([noise_duration, info["original_noise_path"]])
        # org_noise_start = info["noise_start"]
        # noise_duration = info["noise_duration"]
        # if speech_duration > noise_duration:
        #     print(f"Mixture is longer than noise: {key}")
        #     noise_start = 0
        # else:
        #     if speech_duration > (noise_duration - org_noise_start):
        #         noise_start = np.random.randint(0, noise_duration - speech_duration)
        #     else:
        #         noise_start = org_noise_start
        # new_info["noise_start"] = noise_start
        new_metadata[key] = new_info
        sss += 1
        # if sss == 5:
        #     assert len(noise_duration_path_list) == len(new_metadata), (len(noise_duration_path_list), len(new_metadata))
        #     break
    # sort by noise duration
    noise_duration_path_list = sorted(noise_duration_path_list, key=lambda x: x[0], reverse=True)
    # sort metadata with speech duration
    new_metadata = dict(sorted(new_metadata.items(), key=lambda x: x[1]["speech_duration"], reverse=True))
    assert len(noise_duration_path_list) == len(new_metadata), (len(noise_duration_path_list), len(new_metadata))

    # maybe noise_samplerate=16000 and speech_samplerate=8000
    assert noise_samplerate >= speech_samplerate, (noise_samplerate, speech_samplerate)
    coef = noise_samplerate // speech_samplerate

    # re-make noise meatadata
    for i, key in enumerate(new_metadata):
        # print(new_metadata[key]["speech_duration"], noise_duration_path_list[i])
        noise_length, noise_path = noise_duration_path_list[i]
        # when noise is shorter than speech
        new_metadata[key]["original_noise_path"] = noise_path
        speech_length = new_metadata[key]["speech_duration"]
        if (noise_length // coef) <= speech_length:
            # print(f"Mixture is longer than noise: {key}, {speech_length}, {noise_length // coef}")
            new_metadata[key]["noise_start"] = 0
            new_metadata[key]["noise_duration"] = noise_length
        else:
            # print(f"Mixture is shorter than noise: {key}, {speech_length}, {noise_length // coef}")
            new_metadata[key]["noise_start"] = np.random.randint(0, noise_length // coef - speech_length)
            new_metadata[key]["noise_duration"] = speech_length * coef

    with open(args.target_folder / f"simulation_metadata_{spk + 1}spk.json", "w") as f:
        json.dump(new_metadata, f, indent=4)
