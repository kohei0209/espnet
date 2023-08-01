from pathlib import Path
import argparse
import shutil

# noisy rev paths
# audio_names = {
#     1: "1mix_050_050a0504.wav",
#     2: "2mix_050_051_050c010n_0.001892_051c010s_-0.001892.wav",
#     3: "3mix_050_052_421_050a050h_1.0204_052c0108_-1.0204_421o030e_0.wav",
#     4: "4mix_050_053_22h_423_050o020h_2.4117354_053c010a_2.036015_22ha010f_-2.036015_423a010w_-2.4117354.wav",
#     5: "5mix_050_052_053_447_446_050a0514_1.9656248_052c010w_0.74690015_053o020r_-0.74690015_447c0207_-1.9656248_446c020l_0.wav",
# }

# anechoic paths
audio_names = {
    2: "2mix_050_051_050c010n_0.001892_051c010s_-0.001892.wav",
    3: "3mix_050_052_421_050a050h_1.0204_052c0108_-1.0204_421o030e_0.wav",
    4: "4mix_050_052_445_053_050o020j_1.1097069_052o0203_0.81778698_445c0204_-0.81778698_053a0505_-1.1097069.wav",
    5: "5mix_050_420_053_22g_22h_050o020k_0.44480915_420c020m_0.18823178_053o020q_-0.18823178_22go010p_-0.44480915_22ha0102_0.wav",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mix_dir", type=Path, required=True)  # ./data/wsj0_mix
    parser.add_argument("--enh_dir", type=Path, required=True)
    parser.add_argument("--tse_dir", type=Path, required=True)
    parser.add_argument("--n_mix", type=int, nargs="+", required=True)
    args = parser.parse_args()

    tasks = ["enh", "tse"]
    conds = ["without_true_numspk"]

    for n_mix in args.n_mix:
        # mixture
        wav_dir = args.mix_dir / f"{n_mix}speakers" / "wav8k" / "min" / "tt" / "mix"
        if n_mix != 1:
            wav_name = audio_names[n_mix].split("_")[n_mix+1:]
            wav_path = wav_dir / "_".join(wav_name)
        else:
            wav_path = wav_dir / audio_names[n_mix]
        # copy
        output_path = args.output_dir / f"{n_mix}mix" / "mix"
        output_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(wav_path, output_path)

        # reference
        for n in range(1, n_mix + 1):
            # mixture
            wav_dir = args.mix_dir / f"{n_mix}speakers" / "wav8k" / "min" / "tt" / f"s{n}"
            if n_mix != 1:
                wav_name = audio_names[n_mix].split("_")[n_mix+1:]
                wav_path = wav_dir / "_".join(wav_name)
            else:
                wav_path = wav_dir / audio_names[n_mix]
            # copy
            output_path = args.output_dir / f"{n_mix}mix" / "reference" / str(n)
            output_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(wav_path, output_path)

        # separated signals
        for task in tasks:
            for cond in conds:
                for n in range(1, n_mix + 1):
                    wav_path = getattr(args, f"{task}_dir") / task / f"{n_mix}mix" / cond / "logdir" / "output.1" / "wavs" / str(n) / audio_names[n_mix]
                    # wav directory
                    output_path = args.output_dir / f"{n_mix}mix" / task / cond / str(n)
                    output_path.mkdir(parents=True, exist_ok=True)
                    shutil.copy(wav_path, output_path)
