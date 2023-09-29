from pathlib import Path
import glob
import soundfile as sf
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import argparse
import functools

from espnet2.fileio.read_text import read_2columns_text

"""
Script to check whether simulated data is the same as original one
"""


def check_diff(uttID, org_data, new_data):
    org_path = org_data[uttID]
    org, fs1 = sf.read(org_path, dtype="float32")

    new_path = new_data[uttID]
    new, fs2 = sf.read(new_path, dtype="float32")

    assert org_path != new_path, (org_path, new_path)
    assert fs1 == fs2, f"sample rate is different: org {fs1}, new {fs2}"

    diff = abs(org - new).sum()
    if diff != 0.0:
        print(f"{uttID} {abs(org-new).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data", type=Path)
    parser.add_argument("--new_data", type=Path)
    args = parser.parse_args()

    org_data = read_2columns_text(args.original_data / "wav.scp")
    new_data = read_2columns_text(args.new_data / "wav.scp")

    worker = functools.partial(
        check_diff,
        org_data=org_data,
        new_data=new_data,
    )
    keys = list(org_data.keys())

    thread_map(
        worker,
        keys,
        max_workers=32,
    )

    # for uttID in tqdm(org_data):
    #     org_path = org_data[uttID]
    #     org, fs1 = sf.read(org_path, dtype="float32")

    #     new_path = new_data[uttID]
    #     new, fs2 = sf.read(new_path, dtype="float32")

    #     assert org_path != new_path, (org_path, new_path)
    #     assert fs1 == fs2, f"sample rate is different: org {fs1}, new {fs2}"

    #     diff = abs(org - new).sum()
    #     if diff != 0.0:
    #         print(f"{uttID} {abs(org-new).sum()}")
