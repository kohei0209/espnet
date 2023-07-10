import torch
from pathlib import Path

"""
Check whether Separation pre-trained model and TSE fine-tuned model has the same paramters
"""


def check(ss_model_path, tse_model_path):
    ss_params = torch.load(ss_model_path, map_location="cpu")
    tse_params = torch.load(tse_model_path, map_location="cpu")
    for key in tse_params:
        if key in ss_params:
            diff = abs(ss_params[key] - tse_params[key]).sum()
            if diff > 0.:
                # print(f"{key} {diff}")
                k = key.split(".")
                k = ".".join(k[:2])
                print(f"{k}")
        else:
            print(f"Mismatch: {key}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ss_model_path",
        type=Path,
    )
    parser.add_argument(
        "--tse_model_path",
        type=Path,
    )
    args = parser.parse_args()

    check(args.ss_model_path, args.tse_model_path)
