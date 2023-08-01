from pathlib import Path
import argparse
import glob
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

from espnet2.fileio.read_text import read_2columns_text

n_fft = 512
hop_length = 128

"""
python ./local/show_spectrogram.py \
    --wav_dir ./exp/enh_enh_dptnet_eda4thlayer_2-5mix_1bce_raw/enhanced_tt_min_8k/enh/demo \
    --enh_score_dir ./exp/enh_enh_dptnet_eda4thlayer_2-5mix_1bce_raw/enhanced_tt_min_8k \
    --tse_score_dir ./exp/enh_tse_dptnet_eda4thlayer_2-5mix_1bce_sisnr_enhanced_spkselect_raw/enhanced_tt_min_8k/ \
    --n_mix 2
"""


def plot_fig(wav_path, title):
    # load data
    data, fs = sf.read(wav_path, dtype="float32")
    # normalize and stft
    data = data / np.amax(data)
    # plot spectrogram
    fig, ax = plt.subplots(figsize=(6, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=fs, ax=ax, n_fft=n_fft, hop_length=hop_length)
    ax.set(title=title)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Frequency [hz]")
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.subplots_adjust(left=0.125, right=0.95, top=0.88, bottom=0.11)
    plt.savefig(wav_dir / f"{Path(wav_path).stem}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=Path, required=True)
    parser.add_argument("--enh_score_dir", type=Path, required=True)
    parser.add_argument("--tse_score_dir", type=Path, required=True)
    parser.add_argument("--n_mix", type=int, nargs="+", required=True)
    args = parser.parse_args()

    tasks = ["enh", "tse"]
    conds = ["without_true_numspk"]

    for n_mix in args.n_mix:
        # mixture
        wav_dir = args.wav_dir / f"{n_mix}mix" / "mix"
        wav_paths = glob.glob(str(wav_dir) + "/*.wav")
        for wav_path in wav_paths:
            plot_fig(wav_path, "Input mixture")

        # reference
        for n in range(1, n_mix + 1):
            wav_dir = args.wav_dir / f"{n_mix}mix" / "reference" / str(n)
            wav_paths = glob.glob(str(wav_dir) + "/*.wav")
            for wav_path in wav_paths:
                plot_fig(wav_path, f"Reference {n}")

        # separated signals
        for task in tasks:
            for cond in conds:
                for n in range(1, n_mix + 1):
                    # wav directory
                    wav_dir = args.wav_dir / f"{n_mix}mix" / task / cond / str(n)
                    wav_paths = glob.glob(str(wav_dir) + "/*.wav")
                    # score directory
                    score_dir = getattr(args, f"{task}_score_dir") / task / f"{n_mix}mix" / cond / "scoring"
                    scores = read_2columns_text(score_dir / f"SI_SNR_spk{n}")
                    for wav_path in wav_paths:
                        # get key for getting score
                        key = Path(wav_path).stem
                        # SISNR value
                        score = scores[key]
                        title = f"Source {n}: SI-SNR {round(float(score), 2)} [dB]"
                        plot_fig(wav_path, title)
