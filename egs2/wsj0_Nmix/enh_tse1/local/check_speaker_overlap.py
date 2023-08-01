from pathlib import Path
import pprint

from espnet2.fileio.read_text import read_2columns_text


def has_speaker_duplicates(lst):
    # no duplicates if lst has only one compoenent
    if len(lst) <= 1:
        return False

    # remove duplication by changing into set
    unique_set = set(lst)
    # if length is different, there was duplication
    return len(lst) != len(unique_set)


if __name__ == "__main__":
    data_root = Path("./data/")
    stages = ["tr_min_8k", "cv_min_8k", "tt_min_8k"]
    num_duplicates = {}

    # for train, dev, test, check the number of duplications
    for stage in stages:
        num_duplicates[stage] = {}
        wav_scp_path = data_root / stage / "wav.scp"
        wav_scp = read_2columns_text(wav_scp_path)

        for key in wav_scp:
            # key is like 2mix_011_013_011c020h_0.9531_013c020l_-0.9531
            n_mix = int(key[0])
            if n_mix not in num_duplicates[stage]:
                num_duplicates[stage][n_mix] = {"with_duplication": 0, "without_duplication": 0}

            # get speaker ids
            spk_ids = key.split("_")[1:(n_mix + 1)]
            # check whether there is a speaker duplication
            has_duplicate = has_speaker_duplicates(spk_ids)
            # accumulate the results
            if has_duplicate:
                num_duplicates[stage][n_mix]["with_duplication"] += 1
            else:
                num_duplicates[stage][n_mix]["without_duplication"] += 1

    pprint.pprint(num_duplicates)
