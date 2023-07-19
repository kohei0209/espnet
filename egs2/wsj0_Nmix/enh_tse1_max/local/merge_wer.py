import re
from pathlib import Path


def cal_wer(corr, sub, delet, ins):
    return (sub + delet + ins) / (sub + delet + corr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('asr_result_root', type=Path)  # like "exp/enh_enh_dptnet_eda4thlayer_2-5mix_1bce_raw/enhanced_tt_max_8k/enh/5mix/without_true_numspk/whisper"
    parser.add_argument('n_mix', type=int)
    args = parser.parse_args()

    corr, sub, delet, ins = 0, 0, 0, 0
    count = 0
    for spk in range(1, args.n_mix + 1):
        res = args.asr_result_root / f"spk_{spk}" / "score_wer" / "result.txt"
        assert res.exists(), res
        with res.open("r") as f:
            for line in f:
                line = line.strip()
                match = re.fullmatch(r"Scores: \(#C #S #D #I\)\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)$", line)
                if not match:
                    continue
                count += 1
                c, s, d, i = match.groups()
                corr += int(c)
                sub += int(s)
                delet += int(d)
                ins += int(i)
    wer = cal_wer(corr, sub, delet, ins) * 100
    print(f"Counted {count} samples in total, WER is {wer:.3f}%")
