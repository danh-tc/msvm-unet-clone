#!/usr/bin/env python3
"""
plot_robustness.py — Line chart of mean DSC vs corruption severity, one line
per corruption type, from the full-sweep results in results/robustness_*.json.
"""
import json
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS_DIR = osp.join(REPO, "results")
OUT_PATH = osp.join(REPO, "visualizations", "robustness_dsc_vs_severity.png")

CORRUPTIONS = ["gaussian_noise", "poisson_noise", "gaussian_blur", "contrast_shift"]
COLORS = {
    "gaussian_noise": "#1f77b4",
    "poisson_noise": "#ff7f0e",
    "gaussian_blur": "#2ca02c",
    "contrast_shift": "#d62728",
}
LABELS = {
    "gaussian_noise": "Gaussian noise",
    "poisson_noise": "Poisson noise",
    "gaussian_blur": "Gaussian blur",
    "contrast_shift": "Contrast shift",
}


def mean_dsc(cases):
    return float(np.mean([np.mean([m["dsc"] for m in c["per_class"].values()]) for c in cases]))


def main():
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

    baseline = None
    for corruption in CORRUPTIONS:
        path = osp.join(RESULTS_DIR, f"robustness_{corruption}.json")
        with open(path) as fp:
            data = json.load(fp)
        runs = data["runs"]
        severities = sorted(int(s) for s in runs[corruption].keys())
        clean_dsc = mean_dsc(runs["clean"]["0"])
        if baseline is None:
            baseline = clean_dsc
        xs = [0] + severities
        ys = [clean_dsc * 100] + [mean_dsc(runs[corruption][str(s)]) * 100 for s in severities]
        ax.plot(xs, ys, marker="o", linewidth=2, color=COLORS[corruption], label=LABELS[corruption])

    ax.axhline(baseline * 100, color="gray", linestyle="--", linewidth=1, alpha=0.6, label=f"Clean baseline ({baseline*100:.2f}%)")
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("Mean DSC (%)")
    ax.set_title("MSVM-UNet robustness: mean DSC vs corruption severity\n(Synapse test set, 12/12 volumes, frozen checkpoint)")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_ylim(0, 95)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(f"Saved chart to {OUT_PATH}")


if __name__ == "__main__":
    main()
