from pathlib import Path
import glob
import pandas as pd
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm
import argparse
FS_ORIG = 16000

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--wsj0_path", default="../")
parser.add_argument("-o", "--output_folder", default="wsj0-mix")
parser.add_argument("-n", "--n_src", default=2, type=int)
parser.add_argument("-sr", "--samplerate", default=8000, type=int)
parser.add_argument("--len_mode", nargs="+", type=str, default=["min", "max"])
parser.add_argument("--metadata_output_folder", type=Path, default="./metadata2")
parser.add_argument("--scp_output_dir", type=Path, required=True)
args = parser.parse_args()

# Read activlev file. Build {utt_id: activlev} dict
activlev_df = pd.concat([
    pd.read_csv(txt_f, delimiter=" ", names=["utt", "alev"], index_col=False)
    for txt_f in glob.glob("./metadata/activlev/*txt")
])
activlev_dic = dict(zip(list(activlev_df.utt), list(activlev_df.alev)))


def wavwrite_quantize(samples):
    return np.int16(np.round((2 ** 15) * samples))

def wavwrite(file, samples, sr):
    """This is how the old Matlab function wavwrite() quantized to 16 bit.
       We match it here to maintain parity with the original dataset"""
    int_samples = wavwrite_quantize(samples)
    sf.write(file, int_samples, sr, subtype="PCM_16")

# metadata_output_folder = Path("/mnt/kiso-qnap/saijo/universal_se/espnet_use/egs2/wsj0_Nmix/enh1/data/wsj0_mix/scripts/metadata2")
args.metadata_output_folder.mkdir(exist_ok=True)
# scp_output_dir = Path("/mnt/kiso-qnap/saijo/universal_se/espnet_use/egs2/wsj0_Nmix/enh1/data")

for cond in ["tr", "cv", "tt"]:
    # Output folders (wav8k-16k/min-max/tr-cv-tt/mix-src{i})
    base = Path(args.output_folder) / f"{args.n_src}speakers" / f"wav{args.samplerate // 1000}k"
    min_mix_folder =  base / "min"/  cond / "mix"
    min_src_folders = [base / "min"/ cond /  f"s{i+1}" for i in range(args.n_src)]
    max_mix_folder =  base / "max"/  cond / "mix"
    max_src_folders = [base / "max"/ cond /  f"s{i+1}" for i in range(args.n_src)]
    for p in min_src_folders + max_src_folders + [min_mix_folder, max_mix_folder]:
        p.mkdir(parents=True, exist_ok=True)

    # Read SNR scales file
    header = [x for t in zip([f"s_{i}" for i in range(args.n_src)], [f"snr_{i}" for i in range(args.n_src)]) for x in t]
    mix_df = pd.read_csv(f"metadata/mix_{args.n_src}_spk_{cond}.txt", delimiter=" ", names=header, index_col=False)

    # make scp output folder
    scp_output_folder = {}
    for len_mode in args.len_mode:
        scp_output_folder[len_mode] = args.scp_output_dir / f"{cond}_min_{args.samplerate // 1000}k"
        scp_output_folder[len_mode].mkdir(exist_ok=True)

    for idx in tqdm(range(len(mix_df))):

        # Merge filenames for mixture name.  (when mixing weight is 0.450124, it truncates 0.45012, hence the 10x)
        matlab_round = lambda x, y: round(x , y) if abs(x) >= 1.0 else round(x, y + 1)
        pp = lambda x: x.split('/')[-1].replace(".wav", "") if isinstance(x, str) else '{:12.8g}'.format(x).strip()
        filename = "_".join([pp(mix_df[u][idx]) for u in header]) + ".wav"

        for len_mode in args.len_mode:
            mode = "w" if idx == 0 else "a"
            # write the filename of wavs
            with open(args.metadata_output_folder / f"mix_{args.n_src}_spk_{len_mode}_{cond}_mix", mode, encoding="UTF-8") as f:
                fname = filename.replace(".wav", "")
                f.write(fname + "\n")
            # write the path of original files
            for i in range(args.n_src):
                with open(args.metadata_output_folder / f"mix_{args.n_src}_spk_{len_mode}_{cond}_{i}", mode, encoding="UTF-8") as f:
                    p = str(Path(args.wsj0_path) / mix_df[f"s_{i}"][idx])
                    f.write(p + "\n")

            # write scp
            mode = "w" if not (scp_output_folder[len_mode] / "wav_org.scp").exists() else "a"
            folder = min_mix_folder if len_mode == "min" else max_mix_folder
            spk_ids = [] # speaker IDs
            for i in range(args.n_src):
                spk_ids.append(filename.split("_")[i*2][:3])
            spk_ids = '_'.join(spk_ids)
            with open(scp_output_folder[len_mode] / "wav_org.scp", mode, encoding="UTF-8") as f:
                content = f"{args.n_src}mix_{spk_ids}_{fname} {str(folder/filename)}"
                f.write(content + "\n")

            folder = min_src_folders if len_mode == "min" else max_src_folders
            for i in range(1, 6, 1):
                with open(scp_output_folder[len_mode] / f"spk{i}_org.scp", mode, encoding="UTF-8") as f:
                    if args.n_src < i:
                        content = f"{args.n_src}mix_{spk_ids}_{fname} dummy"
                    else:
                        content = f"{args.n_src}mix_{spk_ids}_{fname} {str(folder[i-1]/filename)}"
                    f.write(content + "\n")

            # write utt2spk file
            with open(scp_output_folder[len_mode] / "utt2spk_org", mode, encoding="UTF-8") as f:
                content = f"{args.n_src}mix_{spk_ids}_{fname} {args.n_src}mix_{spk_ids}"
                f.write(content + "\n")