from pathlib import Path
import glob
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm
import argparse
import random
import math
import json
import copy
import pyroomacoustics as pra

FS_ORIG = 16000


def SNR(speech, noise):
    speech_power = ((speech) ** 2).sum()
    noise_power = ((noise) ** 2).sum()
    snr = 10 * np.log10(speech_power / noise_power)
    return snr


parser = argparse.ArgumentParser()
parser.add_argument("--metadata_path", type=Path)
parser.add_argument("--decay_db", type=int, default=60)
args = parser.parse_args()

with open(args.metadata_path) as f:
    metadata = json.load(f)
new_metadata = copy.deepcopy(metadata)

for i, (key, data) in enumerate(tqdm(metadata.items())):
    reverb_speech_paths = data["audio_paths"]["reverberant_speech"]
    noise_path = data["audio_paths"]["noise"]
    rir_paths = data["audio_paths"]["rir"]

    # first compute noise SNR against the mixture
    reverb_speeches = []
    for reverb_speech_path in reverb_speech_paths:
        reverb_speeches.append(sf.read(reverb_speech_path, dtype="float32")[0])
    reverb_mix = np.stack(reverb_speeches, axis=0).sum(axis=0)
    noise = sf.read(noise_path, dtype="float32")[0]
    mixture_snr = SNR(reverb_mix, noise)
    new_metadata[key]["mixture_snr"] = mixture_snr

    # then compute true RT60
    rt60_list = []
    for rir_path in rir_paths:
        rir, fs = sf.read(rir_path, dtype="float32")
        rt60 = pra.experimental.measure_rt60(
            rir, fs=fs, plot=False, decay_db=args.decay_db
        )
        rt60_list.append(rt60)
    new_metadata[key]["measured_rt60"] = rt60_list

# output json file
output_dir = args.metadata_path.parent / "metadata_backup"
output_dir.mkdir(exist_ok=True)
with open(output_dir / args.metadata_path.name, "w") as f:
    json.dump(new_metadata, f, indent=4)
