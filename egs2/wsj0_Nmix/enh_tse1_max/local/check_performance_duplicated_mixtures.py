from pathlib import Path
import pprint
from statistics import mean

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
    data_root = Path(
        "exp/enh_tse_dptnet_eda4thlayer_2-5mix_1bce_sisnr_enhanced_spkselect_raw/enhanced_tt_min_8k/tse/"
    )
    n_mix = [4, 5]
    results = {}

    # for train, dev, test, check the number of duplications
    for nspk in n_mix:
        results[nspk] = {"with_duplication": [], "without_duplication": []}
        score_root = data_root / f"{nspk}mix" / "with_true_numspk" / "scoring"
        scores = []
        for n in range(nspk):
            score_file = score_root / f"SI_SNR_spk{n+1}"
            scores.append(read_2columns_text(score_file))  # key and si_snr
        keys = list(scores[0].keys())

        for key in keys:
            score_tmp = 0
            for n in range(nspk):
                score_tmp += float(scores[n][key])
            score_tmp /= nspk
            # get speaker ids
            spk_ids = key.split("_")[1 : (nspk + 1)]
            # check whether there is a speaker duplication
            has_duplicate = has_speaker_duplicates(spk_ids)

            if has_duplicate:
                results[nspk]["with_duplication"].append(score_tmp)
            else:
                results[nspk]["without_duplication"].append(score_tmp)
        # average
        # print(len(results[nspk]["with_duplication"]), len(results[nspk]["without_duplication"]))
        results[nspk]["all"] = (
            results[nspk]["with_duplication"]
            + results[nspk]["without_duplication"]
        )
        results[nspk]["with_duplication"] = mean(
            results[nspk]["with_duplication"]
        )
        results[nspk]["without_duplication"] = mean(
            results[nspk]["without_duplication"]
        )
        results[nspk]["all"] = mean(results[nspk]["all"])

    pprint.pprint(results)
