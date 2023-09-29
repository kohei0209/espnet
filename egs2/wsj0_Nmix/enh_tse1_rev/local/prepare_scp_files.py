from pathlib import Path
import glob
import soundfile as sf
from tqdm import tqdm
import argparse
from espnet2.fileio.read_text import read_2columns_text
import random
import math
import json

parser = argparse.ArgumentParser()
parser.add_argument("--simulation_metadata_dir", type=Path)
parser.add_argument("--scp_output_dir", type=Path)
parser.add_argument("--num_spk", type=int, default=5)
args = parser.parse_args()


print("Make scp files!!")
metadata = {}
for i in range(args.num_spk):
    with open(args.simulation_metadata_dir / f"simulation_metadata_{i+1}spk.json", "r") as f:
        m = json.load(f)
    original_length, new_length = len(metadata), len(m)
    metadata = dict({**metadata, **m})
    assert len(metadata) == original_length + new_length, "Maybe some keys are duplicated and deleted"
# sort the keys
metadata = sorted(metadata.items())
metadata = dict((key, value) for key, value in metadata)

output_scps = {}
scp_names = ["wav", "noise"]
for i in range(args.num_spk):
    scp_names.append(f"spk{i+1}")
    scp_names.append(f"reverb_spk{i+1}")
    scp_names.append(f"rir_spk{i+1}")
for name in scp_names:
    output_scps[name] = open(args.scp_output_dir / f"{name}.scp", "w")
output_scps["utt2spk"] = open(args.scp_output_dir / "utt2spk", "w")

for audio_id, data in metadata.items():
    # prepare paths
    audio_paths = data["audio_paths"]
    mixture_path = audio_paths["mixture"]
    reverb_speech_paths = audio_paths["reverberant_speech"]
    anechoic_speech_paths = audio_paths["anechoic_speech"]
    rir_paths = audio_paths["rir"]
    noise_path = audio_paths["noise"]
    true_num_spk = len(anechoic_speech_paths)
    assert true_num_spk == int(audio_id[0]), (true_num_spk, audio_id)

    # start writing
    output_scps["wav"].write(f"{audio_id} {mixture_path}\n")
    output_scps["noise"].write(f"{audio_id} {noise_path}\n")
    for i in range(args.num_spk):
        if i < true_num_spk:
            output_scps[f"spk{i+1}"].write(f"{audio_id} {anechoic_speech_paths[i]}\n")
            output_scps[f"reverb_spk{i+1}"].write(f"{audio_id} {reverb_speech_paths[i]}\n")
            output_scps[f"rir_spk{i+1}"].write(f"{audio_id} {rir_paths[i]}\n")
        else:
            output_scps[f"spk{i+1}"].write(f"{audio_id} dummy\n")
            output_scps[f"reverb_spk{i+1}"].write(f"{audio_id} dummy\n")
            output_scps[f"rir_spk{i+1}"].write(f"{audio_id} dummy\n")
    # utt2spk file
    speaker_ids = audio_id.split("_")[:(true_num_spk+1)]
    speaker_ids = "_".join(speaker_ids)
    output_scps["utt2spk"].write(f"{audio_id} {speaker_ids}\n")

for name in scp_names:
    output_scps[name].close()