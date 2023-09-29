from pathlib import Path
import glob
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from functools import partial
import argparse
from espnet2.fileio.read_text import read_2columns_text


def wavwrite_quantize(samples):
    return np.int16(np.round((2**15) * samples))


def wavwrite(file, samples, sr):
    """This is how the old Matlab function wavwrite() quantized to 16 bit.
    We match it here to maintain parity with the original dataset"""
    int_samples = wavwrite_quantize(samples)
    sf.write(file, int_samples, sr, subtype="PCM_16")


def worker(wav_path, mix_output_folder, speech_output_folder):
    wav_path = Path(wav_path)
    spkID = wav_path.parent.name
    filename = wav_path.stem
    uttID = f"1mix_{spkID}_{filename}"

    # load wavs and resample
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if sr > args.samplerate:
        wav = resample_poly(wav, args.samplerate, sr)
    wav = wav / np.amax(wav) * 0.9

    # save the audio
    mix_output_path = mix_output_folder / f"{uttID}.wav"
    wavwrite(str(mix_output_path), wav, sr=args.samplerate)

    # In 1-mix, mixture and ground truth is exactly the same
    # So we make symbolic link to mixture instead of saving audio
    (speech_output_folder / f"{uttID}.wav").symlink_to(mix_output_path)


parser = argparse.ArgumentParser()
parser.add_argument("--wsj0_metadata_folder", type=Path)
parser.add_argument("--wav_output_folder", type=Path)
parser.add_argument("--len_mode", default="min", type=str)
parser.add_argument("-sr", "--samplerate", default=8000, type=int)
args = parser.parse_args()

len_mode = args.len_mode
samplerate_str = str(args.samplerate // 1000) + "k"
args.wav_output_folder = (
    args.wav_output_folder / f"wav{samplerate_str}" / len_mode
)

for cond in ["tr", "cv", "tt"]:
    wav_output_folder = args.wav_output_folder / cond
    mix_output_folder = wav_output_folder / "mix"
    speech_output_folder = wav_output_folder / "s1"
    mix_output_folder.mkdir(exist_ok=True, parents=True)
    speech_output_folder.mkdir(exist_ok=True)
    wav_paths = []
    for n in range(5):
        metadata_path = (
            args.wsj0_metadata_folder / f"mix_5_spk_{len_mode}_{cond}_{n}"
        )
        with open(metadata_path, "r") as f:
            metadata = f.read().split("\n")[:-1]
        for path in metadata:
            if path not in wav_paths:
                wav_paths.append(path)
    print(f"{cond}, {len(wav_paths)}")

    worker2 = partial(
        worker,
        mix_output_folder=mix_output_folder,
        speech_output_folder=speech_output_folder,
    )
    thread_map(
        worker2,
        wav_paths,
        max_workers=32,
    )

    """
    for wav_path in tqdm(wav_paths):
        wav_path = Path(wav_path)
        spkID = wav_path.parent.name
        filename = wav_path.stem
        uttID = f"1mix_{spkID}_{filename}"

        # load wavs and resample
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if sr > args.samplerate:
            wav = resample_poly(wav, args.samplerate, sr)
        wav = wav / np.amax(wav) * 0.9

        # save the audio
        wavwrite(str(mix_output_folder / f"{uttID}.wav"), wav, sr=args.samplerate)

        # In 1-mix, mixture and ground truth is exactly the same
        # So we make symbolic link to mixture instead of saving audio
        (speech_output_folder / f"{uttID}.wav").symlink_to(mix_output_path)
        # wavwrite(str(speech_output_folder / f"{uttID}.wav"), wav, sr=args.samplerate)
    """

'''
from pathlib import Path
import glob
import pandas as pd
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm
import argparse
from espnet2.fileio.read_text import read_2columns_text

FS_ORIG = 16000

num_data = {"tr": 20000, "cv": 5000, "tt": 3000}

parser = argparse.ArgumentParser()
parser.add_argument("--wsj0_metadata_folder", type=Path)
parser.add_argument("--wav_output_folder", type=Path)
parser.add_argument("--scp_output_folder", type=Path)
parser.add_argument("--len_mode", default="min", type=str)
parser.add_argument("-sr", "--samplerate", default=8000, type=int)
args = parser.parse_args()


def wavwrite_quantize(samples):
    return np.int16(np.round((2**15) * samples))


def wavwrite(file, samples, sr):
    """This is how the old Matlab function wavwrite() quantized to 16 bit.
    We match it here to maintain parity with the original dataset"""
    int_samples = wavwrite_quantize(samples)
    sf.write(file, int_samples, sr, subtype="PCM_16")


len_mode = args.len_mode
samplerate_str = str(args.samplerate // 1000) + "k"
args.wav_output_folder = (
    args.wav_output_folder / f"wav{samplerate_str}" / len_mode
)

for cond in ["tr", "cv", "tt"]:
    wav_output_folder = args.wav_output_folder / cond
    mix_output_folder = wav_output_folder / "mix"
    speech_output_folder = wav_output_folder / "s1"
    mix_output_folder.mkdir(exist_ok=True, parents=True)
    speech_output_folder.mkdir(exist_ok=True)
    wav_paths = []
    for n in range(5):
        metadata_path = (
            args.wsj0_metadata_folder / f"mix_5_spk_{len_mode}_{cond}_{n}"
        )
        with open(metadata_path, "r") as f:
            metadata = f.read().split("\n")[:-1]
        for path in metadata:
            if path not in wav_paths:
                wav_paths.append(path)
    print(f"{cond}, {len(wav_paths)}")

    mix_scp = open(
        args.scp_output_folder
        / f"{cond}_{len_mode}_{samplerate_str}"
        / "tmp_1mix.scp",
        "w",
    )
    speech_scps = []
    for n in range(1, 6):
        speech_scps.append(
            open(
                args.scp_output_folder
                / f"{cond}_{len_mode}_{samplerate_str}"
                / f"tmp_1mix_spk{n}.scp",
                "w",
            )
        )
    for i, wav_path in enumerate(tqdm(wav_paths)):
        wav_path = Path(wav_path)
        spkID = wav_path.parent.name
        filename = wav_path.stem
        uttID = f"1mix_{spkID}_{filename}"

        # load wavs and resample
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if sr > args.samplerate:
            wav = resample_poly(wav, args.samplerate, sr)
        wav = wav / np.amax(wav) * 0.9

        # write scp files
        mix_output_path = str(mix_output_folder / f"{uttID}.wav")
        mix_scp.write(f"{uttID} {mix_output_path}\n")
        speech_output_path = str(speech_output_folder / f"{uttID}.wav")
        speech_scps[0].write(f"{uttID} {speech_output_path}\n")
        for n in range(2, 6):
            speech_scps[n - 1].write(f"{uttID} dummy\n")

        # save the audio
        wavwrite(mix_output_path, wav, sr=args.samplerate)

        # In 1-mix, mixture and ground truth is the same
        # We make symbolic link to mixture instead of saving audio
        speech_output_path.symlink_to(mix_output_path)
        # wavwrite(speech_output_path, wav, sr=args.samplerate)

    # close files
    mix_scp.close()
    for speech_scp in speech_scps:
        speech_scp.close()
'''
