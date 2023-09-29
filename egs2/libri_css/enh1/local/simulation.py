import numpy as np
import random
import pyroomacoustics as pra
import argparse
from pathlib import Path
import json
import copy
import soundfile as sf
from tqdm import tqdm
import fast_bss_eval
import threading
import concurrent.futures
from itertools import repeat
from scipy.signal import resample_poly


def simulate_src(
    dry_source,
    room_size,
    src_position,
    sensor_position,
    rt60,
    seed,
    fs=8000,
    use_rand_ism=True,
):
    # print(room_size.shape, src_position.shape, sensor_position.shape)
    random.seed(seed)
    np.random.seed(seed)

    e_absorption, max_order = pra.inverse_sabine(rt60, room_size)
    # print(e_absorption, max_order, rt60, room_size)

    room = pra.ShoeBox(
        room_size,
        fs=fs,
        use_rand_ism=use_rand_ism,
        max_rand_disp=0.08,
        max_order=max_order,
        materials=pra.Material(e_absorption),
    )
    room.add_microphone_array(pra.MicrophoneArray(sensor_position, fs=room.fs))
    room.add_source(src_position, signal=dry_source)
    room.simulate(snr=None)
    reverberant_source = room.mic_array.signals.copy()
    rir = room.rir.copy()

    random.seed(seed)
    np.random.seed(seed)

    anechoic_room = pra.ShoeBox(
        room_size,
        fs=fs,
        use_rand_ism=use_rand_ism,
        max_rand_disp=0.08,
        max_order=0,
        materials=pra.Material(e_absorption),
    )
    # anechoic_room = pra.ShoeBox(room_size, fs=fs, max_order=0)
    anechoic_room.add_microphone_array(
        pra.MicrophoneArray(sensor_position, fs=anechoic_room.fs)
    )
    anechoic_room.add_source(src_position, signal=dry_source)
    # use first 50ms of RIR for clean signal
    rir_50ms = (rir[0][0])[: fs // 20]
    anechoic_room.rir = [[rir_50ms]]
    # simulation
    anechoic_room.simulate(snr=None)
    anechoic_source = anechoic_room.mic_array.signals.copy()

    return reverberant_source, anechoic_source, rir


def adjust_SNR(speech, noise, snr):
    speech_power = np.mean(np.square(speech))
    noise_power = np.mean(np.square(noise))
    alpha = np.sqrt((noise_power / speech_power) * (10 ** (snr / 10)))

    return alpha


def SNR(speech, noise):
    speech_power = ((speech) ** 2).sum()
    noise_power = ((noise) ** 2).sum()
    snr = 10 * np.log10(speech_power / noise_power)
    return snr


def wavwrite_quantize(samples):
    return np.int16(np.round((2**15) * samples))


def wavwrite(file, samples, sr):
    """This is how the old Matlab function wavwrite() quantized to 16 bit.
    We match it here to maintain parity with the original dataset"""
    int_samples = wavwrite_quantize(samples)
    sf.write(file, int_samples, sr, subtype="PCM_16")


def run(
    mixinfo,
):
    reverbs, anechoics, rirs = [], [], []

    for s, original_source_path in enumerate(mixinfo["drysrc_paths"]):
        drysrc, fs = sf.read(original_source_path)
        num_samples = mixinfo["speech_duration"]
        assert drysrc.shape[0] == num_samples, (drysrc.shape[0], num_samples)
        reverb, anechoic, rir = simulate_src(
            drysrc,
            np.array(mixinfo["room_size"]),
            np.array(mixinfo["speaker_positions"])[s].T,
            np.array(mixinfo["microphone_position"])[..., None],
            mixinfo["rt60"],
            mixinfo["seed"],
            fs=fs,
        )
        # print("sisdr", fast_bss_eval.si_sdr(reverb[:, :num_samples], anechoic[:, :num_samples]))
        # print("sdr", fast_bss_eval.sdr(reverb[:, :num_samples], anechoic[:, :num_samples]))

        reverbs.append(reverb[:, :num_samples])
        anechoics.append(anechoic[:, :num_samples])
        rirs.append(np.array(rir[0]))

    reverbs, anechoics = np.array(reverbs), np.array(anechoics)

    # noise preparation
    noise, fs_noise = sf.read(
        mixinfo["original_noise_path"],
        start=mixinfo["noise_start"],
        frames=mixinfo["noise_duration"],
    )
    if noise.ndim == 2:
        noise = noise[..., -1]  # select first channel
    if fs_noise != fs:
        noise = resample_poly(noise, fs, fs_noise)
    # pad noise if noise is shorter than speech
    if noise.shape[0] < drysrc.shape[0]:
        noise = np.pad(
            noise, (0, drysrc.shape[0] - noise.shape[0]), mode="wrap"
        )
    noise = np.array(noise)
    noise = noise / np.max(noise)
    assert drysrc.shape[0] == noise.shape[0], (drysrc.shape, noise.shape)
    # noise = noise[None]

    # adjust noise snr based on the strongest speech, instead of mixture
    speech_powers = (abs(reverbs) ** 2).sum(axis=-1)
    inactive_source_idx = np.argmin(speech_powers)
    weight = adjust_SNR(
        reverbs[inactive_source_idx], noise, snr=mixinfo["noise_snr"]
    )
    reverbs *= weight
    anechoics *= weight
    mixture = np.sum(reverbs, axis=0) + noise
    # print(speech_powers, inactive_source_idx, weight)
    # print(mixinfo['noise_snr'], SNR(reverbs[inactive_source_idx], noise), SNR(np.sum(reverbs, axis=0), noise))

    # to make audios less than 1
    coef = abs(mixture).max() / 0.95
    mixture = mixture / coef
    reverbs = reverbs / coef
    anechoics = anechoics / coef
    noise = noise / coef
    # print(mixture.shape, noise.shape, reverbs.shape, anechoics.shape)
    # print("sisdr", fast_bss_eval.si_sdr(anechoics[:, 0], np.tile(mixture, (anechoics.shape[0],1))).mean())
    # print("sdr", fast_bss_eval.sdr(anechoics[:, 0], np.tile(mixture, (anechoics.shape[0],1))).mean())

    audio_paths = mixinfo["audio_paths"]
    wavwrite(audio_paths["mixture"], mixture.T, fs)
    wavwrite(audio_paths["noise"], noise, fs)
    for j in range(reverbs.shape[0]):
        wavwrite(audio_paths["reverberant_speech"][j], reverbs[j].T, fs)
        wavwrite(audio_paths["anechoic_speech"][j], anechoics[j].T, fs)
        wavwrite(audio_paths["rir"][j], rirs[j].T, fs)


def make_src_data(args):
    random.seed(0)
    np.random.seed(0)

    with open(args.metadata_path) as f:
        mixinfo_dict = json.load(f)
    mix_info = []
    for key, value in mixinfo_dict.items():
        mix_info.append(value)

    n_thread = args.n_thread
    # mix_info = mix_info[:10]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_thread
    ) as excuter:
        list(
            tqdm(
                excuter.map(
                    run,
                    mix_info,
                ),
                total=len(mix_info),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=Path)
    parser.add_argument("-nj", "--n_thread", type=int, default=8)
    args = parser.parse_args()
    make_src_data(args)
