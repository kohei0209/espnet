from pathlib import Path
import argparse
import json

from espnet2.fileio.read_text import read_2columns_text

"""
This code is for rewrite scp files with replacing the specified words with other words.
For example, this code can replace "min" to "max" in the path.
It is useful to make scp files of "max" version of WSJ from "min" version without changing other configuration.

Source: ./data/wsj0_mix/5speakers/wav8k/min/tt/s5/442o030e_1.0120967_420o030c_0.9186296_053o020p_-0.9186296_443c020l_-1.0120967_050c010j_0.wav
Target: ./data/wsj0_mix/5speakers/wav8k/max/tt/s5/442o030e_1.0120967_420o030c_0.9186296_053o020p_-0.9186296_443c020l_-1.0120967_050c010j_0.wav

python ./local/copy_enroll_files.py \
    --source_folder ../enh_tse1/data/tt_min_8k --target_folder ./data/tt_max_8k \
    --source_words /min/ /enh_tse1/ --target_words /max/ /enh_tse1_max/ --num_spk 5
"""

parser = argparse.ArgumentParser()
parser.add_argument("--source_folder", type=Path, required=True)
parser.add_argument("--target_folder", type=Path, required=True)
parser.add_argument(
    "--source_words",
    nargs="*",
    type=str,
    required=True,
    help="Specify the word that you would like to replace from",
)
parser.add_argument(
    "--target_words",
    nargs="*",
    type=str,
    required=True,
    help="Specify the word that you would like to replace with",
)
parser.add_argument("--num_spk", type=int, default=5)
args = parser.parse_args()


assert len(args.source_words) == len(args.target_words)

for spk in range(args.num_spk):
    filename = f"simulation_metadata_{spk + 1}spk.json"
    if (args.source_folder / filename).exists():
        print(f"Copy {args.source_folder / filename} while changing paths")
        with open(args.source_folder / filename, "r") as f:
            source_file = json.load(f)
        # rewrite some specified words
        for utt_id, data in source_file.items():
            for audio_type, paths in data["audio_paths"].items():
                if isinstance(paths, str):
                    for source_word, target_word in zip(
                        args.source_words, args.target_words
                    ):
                        paths = paths.replace(source_word, target_word)
                    source_file[utt_id]["audio_paths"][audio_type] = paths
                else:
                    for i, path in enumerate(paths):
                        for source_word, target_word in zip(
                            args.source_words, args.target_words
                        ):
                            path = path.replace(source_word, target_word)
                        source_file[utt_id]["audio_paths"][audio_type][
                            i
                        ] = path
            for i, path in enumerate(data["drysrc_paths"]):
                for source_word, target_word in zip(
                    args.source_words, args.target_words
                ):
                    path = path.replace(source_word, target_word)
                source_file[utt_id]["drysrc_paths"][i] = path
            # original noise path
            path = data["original_noise_path"]
            for source_word, target_word in zip(
                args.source_words, args.target_words
            ):
                path = path.replace(source_word, target_word)
            source_file[utt_id]["original_noise_path"] = path
    else:
        print(f"{args.source_folder / filename} does not exist")

    # save a new json file to target folder
    with open(args.target_folder / filename, "w") as f:
        json.dump(source_file, f, indent=4)
