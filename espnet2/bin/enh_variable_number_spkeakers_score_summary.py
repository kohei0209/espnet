#!/usr/bin/env python3
import argparse
import logging
import sys
from itertools import chain
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union
import re
import json

import numpy as np
import torch
import yaml
from tqdm import trange
from typeguard import check_argument_types

from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.read_text import read_2columns_text

def score_summary(args):
    # get scores for all number of speakers
    # write results to some files
    results = {}
    for protocol in args.protocols:
        if protocol == "Est_num_spk":
            scores = read_2columns_text(args.score_dir / f"{protocol}")
            results[protocol] = speaker_count(scores, args.max_num_spk)
        else:
            scores = []
            for spk in range(1, args.max_num_spk+1):
                score = read_2columns_text(args.score_dir / f"{protocol}_spk{spk}")
                scores.append(score)
            results[protocol] = get_score(scores, args.max_num_spk)
    return results

def get_score(scores, max_num_spk):
    # get average score for speech metrics
    score_total, count = {}, {}
    for score in scores:
        for key, value in score.items():
            # first word of key must be like 2mix, 3mix, ...
            nmix = int(key[0])
            assert nmix <= max_num_spk, (nmix, max_num_spk)
            if value == "dummy":
                continue
            if nmix not in score_total.keys():
                score_total[nmix] = float(value)
                count[nmix] = 1
            else:
                score_total[nmix] += float(value)
                count[nmix] += 1
    for nmix in score_total:
        score_total[nmix] = str(round(score_total[nmix]/count[nmix], 4))
    return score_total

def speaker_count(score, max_num_spk):
    # speaker counting accuracy
    # under, over estimation
    speaker_count = {}
    for key, est_num_spk in score.items():
        # first word of key must be like 2mix, 3mix, ...
        nmix = int(key[0])
        if nmix not in speaker_count.keys():
            speaker_count[nmix] = {}
        if est_num_spk not in speaker_count[nmix].keys():
            speaker_count[nmix][est_num_spk] = 1
        else:
            speaker_count[nmix][est_num_spk] += 1
    # compute percentage
    for nmix, counts in speaker_count.items():
        total = sum(list(counts.values()))
        for est_num_spk, count in counts.items():
            speaker_count[nmix][est_num_spk] = f"{count}  {round(100*count/total, 2)}[%]"
    return speaker_count

def write_results(output_dir, results, max_num_spk):
    for protocol, scores in results.items():
        for nmix, score in scores.items():
            output_dir_nmix = output_dir / f"{nmix}mix_results"
            output_dir_nmix.mkdir(exist_ok=True)
            if protocol != "Est_num_spk":
                with open(output_dir_nmix / f"{protocol}.txt", "w") as f:
                    f.write(score)
            else:
                with open(output_dir_nmix / f"{protocol}.json", "w") as f:
                    json.dump(score, f)
    with open(output_dir / f"score_summary.json", "w") as f:
        json.dump(results, f, indent=4)

def get_parser():
    parser = config_argparse.ArgumentParser()
    parser.add_argument("--score_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--protocols", type=str, required=True)
    parser.add_argument("--max_num_spk", type=int, required=True)

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.protocols = args.protocols.split(" ")
    # args.protocols.remove("Est_num_spk")
    results = score_summary(args)
    write_results(args.output_dir, results, args.max_num_spk)

if __name__ == "__main__":
    main()