from pathlib import Path
import argparse
import yaml
import torch


"""
Copy enh model parameter to tse model.
This code was used when enh model is finetuned with max version but tse model is not finetuned.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--enh_model_path", type=Path, required=True)
parser.add_argument("--tse_model_path", type=Path, required=True)
parser.add_argument("--output_suffix", type=str, required=True)
args = parser.parse_args()

enh_params = torch.load(args.enh_model_path)
tse_params = torch.load(args.tse_model_path)
with open(args.tse_model_path.parent / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

enh_module_names = config["freeze_param"]
print(f"These modules are integrated:\n{enh_module_names}")
integrated_module_names, frozen_module_names = [], []
for t in enh_module_names:
    for k, p in tse_params.items():
        if k.startswith(t + ".") or k == t:
            # tse_params[k] = enh_params[k]
            if k not in integrated_module_names:
                integrated_module_names.append(k)
        # else:
        #     if k not in frozen_module_names:
        #         frozen_module_names.append(k)

print(integrated_module_names)
for name in integrated_module_names:
    tse_params[name] = enh_params[name]

torch.save(tse_params, args.tse_model_path.parent / args.output_suffix)
