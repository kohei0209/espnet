from pathlib import Path
import argparse
import json
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", type=Path, required=True)
args = parser.parse_args()

results_dirs = glob.glob(str(args.exp_dir) + "/*mix")
num_spk_list = [Path(results_dir).name[0] for results_dir in results_dirs]
confusion_matrix = np.zeros([len(num_spk_list), len(num_spk_list)])
for i, results_dir in enumerate(results_dirs):
    root = Path(results_dir)
    with open(Path(results_dir) / "without_true_numspk" / "score_summary.json", "r") as f:
        result = json.load(f)
    est_num_spk = result["Est_num_spk"][num_spk_list[i]]
    for j, num_spk in enumerate(num_spk_list):
        if num_spk in est_num_spk:
            est_rate = est_num_spk[num_spk]  # this is string like "3000  100.0[%]"
            est_rate = float(est_rate.split(" ")[-1][:-3])  # now it's like 100.0
            confusion_matrix[i][j] = est_rate

ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='.2f', vmin=0.0, cbar=False, square=True)
for text in ax.texts:
    text.set_fontsize(16)
ax.set_title("Speaker counting accuracy [%]", fontsize=18)
ax.set_xlabel("Estimated number of speakers", fontsize=16)
ax.set_xticklabels(num_spk_list)
ax.set_ylabel("Oracle number of speakers", fontsize=16)
ax.set_yticklabels(num_spk_list)
ax.tick_params(axis='both', which='major', labelsize=16)

plt.subplots_adjust(bottom=0.15)
plt.savefig(args.exp_dir / "confusion_marix.pdf")
