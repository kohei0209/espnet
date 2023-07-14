from pathlib import Path
import glob
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm
import argparse
from espnet2.fileio.read_text import read_2columns_text
import random
import math
import json
FS_ORIG = 16000

random.seed(0)
np.random.seed(0)
# config = {
#     "snr_range": [-5, 15],
#     "rt60": [0.2, 0.6],
#     "room_size": [[5., 10.], [5., 10.], [3., 4.],],
# }

categories = ["1mix", "2mix", "3mix", "4mix", "5mix"]

parser = argparse.ArgumentParser()
parser.add_argument("--wav_scp_folder", default="wsj0-mix", type=Path)
parser.add_argument("--wham_folder", default="wsj0-mix", type=Path)
parser.add_argument("--wav_output_folder", default="wsj0-mix", type=Path)
parser.add_argument("--simulation_config_path", type=Path)
parser.add_argument("--len_mode", type=str, default="min")
parser.add_argument("-sr", "--samplerate", default=8000, type=int)
args = parser.parse_args()

def get_3d_position(x_range, y_range, z_range):
    x = random.uniform(x_range[0], x_range[1])
    y = random.uniform(y_range[0], y_range[1])
    z = random.uniform(z_range[0], z_range[1])
    return [x, y, z]

def check_distance(pos_target, pos_others, thres_distance=0.5):
    # Return True if pos_target is closer to other sources than thres_distance
    # note: z dimension is ignored,
    flag = False
    for pos_other in pos_others:
        distance = math.sqrt((pos_target[0]-pos_other[0])**2 + (pos_target[1]-pos_other[1])**2)
        flag = flag or (distance < thres_distance)
    return flag

samplerate_str = str(args.samplerate // 1000) + "k"
len_mode = args.len_mode
with open(args.simulation_config_path, "r") as f:
    config = json.load(f)
print(config)

# for cond in ["tr", "cv", "tt"]:
#     for i in range(5):
#         base_dir = args.wav_output_folder / f"{i+1}speakers" / f"wav_{samplerate_str}" / len_mode / cond
#         (base_dir / "mix").mkdir(exist_ok=True, parents=True)
#         (base_dir / "noise").mkdir(exist_ok=True)
#         for n in range(i+1):
#             (base_dir / f"reverb_s{n+1}").mkdir(exist_ok=True)
#             (base_dir / f"s{n+1}").mkdir(exist_ok=True)
#             (base_dir / f"rir_s{n+1}").mkdir(exist_ok=True)


for cond in ["tt"]:
    # speech related
    scp_path = args.wav_scp_folder / f"{cond}_{len_mode}_{samplerate_str}"
    mix_scp = read_2columns_text(scp_path / "wav.scp")
    speech_scps = []
    for i in range(1, 6, 1):
        speech_scps.append(read_2columns_text(scp_path / f"spk{i}.scp"))
    # noise related
    wham_dir = args.wham_folder / f"{cond}"
    noise_info = {}
    for noise_path in tqdm(list(wham_dir.iterdir())):
        # avoid speech purtabated audios in librimix
        if "sp" in noise_path.name[-8:]:
            continue
        noise_path = str(noise_path)
        noise_len = sf.info(noise_path).frames // (FS_ORIG//args.samplerate)
        noise_info[noise_path] = noise_len
    noise_info = sorted(noise_info.items(), key=lambda x:x[1], reverse=True)

    # get simulation metadata
    for category in categories:
        print(f"Start preparing {category} mixture's simulation metadata")
        n_spk = int(category[0])
        mix_info = {}
        metadata = {}
        for id, path in tqdm(mix_scp.items()):
            if id[:4] != category:
                continue
            mix_len = sf.info(path).frames
            mix_info[id] = mix_len
        mix_info = sorted(mix_info.items(), key=lambda x:x[1], reverse=True)
        assert len(mix_info) <= len(noise_info), (len(mix_info), len(noise_info))

        print(len(mix_info), len(noise_info))
        for i in range(len(mix_info)):
            noise_len = noise_info[i][-1]
            mix_len = mix_info[i][-1]
            offset = noise_len - mix_len
            assert offset >= 0, (mix_info[i][-1], noise_info[i][-1])
            key = mix_info[i][0]

            # drysource paths
            drysrc_paths = []
            for n in range(n_spk):
                drysrc_paths.append(speech_scps[n][key])
            # output paths
            base_dir = args.wav_output_folder / f"{n_spk}speakers" / f"wav_{samplerate_str}" / len_mode / cond
            filename = Path(drysrc_paths[0]).name # *.wav
            mixture_path = str(base_dir / "mix" / filename)
            (base_dir / "mix").mkdir(exist_ok=True, parents=True)
            noise_path = str(base_dir / "noise" / filename)
            (base_dir / "noise").mkdir(exist_ok=True)
            reverb_speech_paths, anechoic_speech_paths, rir_paths = [], [], []
            for n in range(n_spk):
                reverb_speech_paths.append(str(base_dir / f"reverb_s{n+1}" / filename))
                anechoic_speech_paths.append(str(base_dir / f"s{n+1}" / filename))
                rir_paths.append(str(base_dir / f"rir_s{n+1}" / filename))
                (base_dir / f"reverb_s{n+1}").mkdir(exist_ok=True)
                (base_dir / f"s{n+1}").mkdir(exist_ok=True)
                (base_dir / f"rir_s{n+1}").mkdir(exist_ok=True)

            ## randomly set the parameters ##
            start = random.randint(0, offset-1)
            duration = mix_len
            # noise SNR
            snr = random.uniform(config["snr_range"][0], config["snr_range"][1])
            # reverberation time
            rt60 = random.uniform(config["rt60"][0], config["rt60"][1])
            # room size
            room_size = get_3d_position(config["room_size"][0], config["room_size"][1], config["room_size"][2])
            # microphone position (single channel)
            mic_position = get_3d_position(
                [room_size[0]//2-1, room_size[0]//2+1],
                [room_size[1]//2-1, room_size[1]//2+1],
                [1, 2],
            )
            # speaker positions
            speaker_positions = []
            for s in range(n_spk):
                speaker_position = get_3d_position(
                    [0.5, room_size[0]-0.5],
                    [0.5, room_size[1]-0.5],
                    [1, 2],
                )
                flag = True
                while check_distance(speaker_position, speaker_positions):
                    speaker_position = get_3d_position(
                        [0.5, room_size[0]-0.5],
                        [0.5, room_size[1]-0.5],
                        [1, 2],
                    )
                speaker_positions.append(speaker_position)
            # seed for parallel simulation
            seed = random.randint(0, 2**32-1)
            metadata[key] = {
                "audio_paths": {
                    "mixture": mixture_path,
                    "reverberant_speech": reverb_speech_paths,
                    "anechoic_speech": anechoic_speech_paths,
                    "rir": rir_paths,
                    "noise": noise_path,
                },
                "drysrc_paths": drysrc_paths,
                "speech_duration": mix_len,
                "original_noise_path": noise_info[i][0],
                "noise_snr": snr,
                "noise_start": start * (FS_ORIG//args.samplerate),
                "noise_duration": mix_len * (FS_ORIG//args.samplerate),
                "room_size": room_size,
                "microphone_position": mic_position,
                "speaker_positions": speaker_positions,
                "rt60": rt60,
                "seed": seed,
            }
            # if i==3:
            #     break

        with open(args.wav_scp_folder / f"{cond}_{len_mode}_{samplerate_str}" / f"simulation_metadata_{n_spk}spk.json", "w") as f:
            json.dump(metadata, f, indent=4)
