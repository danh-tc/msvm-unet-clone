"""
infer_robustness.py — Zero-shot robustness sweep for MSVM-UNet.

Frozen checkpoint, corrupt the input CT slices (gaussian_noise, poisson_noise,
gaussian_blur, contrast_shift — see corruptions.py) at increasing severity,
measure DSC/HD95 degradation vs the clean baseline. See ROBUSTNESS_EVAL_PLAN.md.

Usage:
    # smoke test: 1 volume, clean only (severity=0, sanity check vs baseline)
    python infer_robustness.py data/Synapse/test_vol_h5/case0001.npy.h5 --severities 0

    # smoke test: 1 volume, all corruptions, mild + harshest severity only
    python infer_robustness.py data/Synapse/test_vol_h5/case0001.npy.h5 --severities 1 4

    # full sweep, all 12 test cases
    python infer_robustness.py --all
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch

REPO = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, REPO)

from infer_single import (
    CKPT_DEFAULT,
    CLASS_NAMES,
    TEST_LIST_DEFAULT,
    TEST_VOL_DIR_DEFAULT,
    calc_dsc_hd95,
    load_model,
    load_volume,
    predict_volume,
)
from corruptions import CORRUPTIONS, MAX_SEVERITY, corrupt_volume

OUTPUT_JSON_DEFAULT = osp.join(REPO, "results", "infer_robustness_results.json")


def compute_metrics(prediction: np.ndarray, label: np.ndarray) -> dict:
    per_class = {}
    for c, name in enumerate(CLASS_NAMES, start=1):
        dsc, hd95 = calc_dsc_hd95(prediction == c, label == c)
        per_class[name] = {"dsc": dsc, "hd95": hd95}
    return per_class


def mean_over_classes(per_class: dict, key: str) -> float:
    return float(np.mean([m[key] for m in per_class.values()]))


def save_results(output_json: str, runs: dict, args, severities: list) -> None:
    """Write results so far as JSON, atomically (tmp file + rename) so a kill
    mid-write never leaves a truncated/corrupt file. Called after every
    corruption/severity block finishes, not just once at the end, so a crash
    or OOM partway through a long sweep doesn't lose already-computed runs.
    """
    os.makedirs(osp.dirname(output_json), exist_ok=True)
    payload = {
        "config": {
            "ckpt": args.ckpt, "corruptions": args.corruptions,
            "severities": severities, "seed": args.seed,
        },
        "runs": runs,
    }
    tmp_path = output_json + ".tmp"
    with open(tmp_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    os.replace(tmp_path, output_json)


def evaluate_volume(
    volume_path: str, model, device, corruption: str, severity: int, seed: int,
) -> dict:
    case_name = osp.basename(volume_path)
    image, label = load_volume(volume_path)
    if severity > 0:
        image = corrupt_volume(image, corruption, severity, seed=seed)
    prediction = predict_volume(model, image, str(device))
    per_class = compute_metrics(prediction, label)
    return {"case_name": case_name, "per_class": per_class}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("volume", nargs="?", help="Path to a single .npy.h5 test volume")
    parser.add_argument(
        "--all", action="store_true",
        help="Run over the full test set (lists/lists_Synapse/test.txt) instead of one volume",
    )
    parser.add_argument("--ckpt", default=CKPT_DEFAULT, help="MSVM-UNet checkpoint path")
    parser.add_argument(
        "--corruptions", nargs="+", default=list(CORRUPTIONS),
        choices=list(CORRUPTIONS), help="Which corruption types to sweep",
    )
    parser.add_argument(
        "--severities", nargs="+", type=int, default=list(range(MAX_SEVERITY + 1)),
        help=f"Severity levels to sweep (0..{MAX_SEVERITY}, 0 = clean)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for corruption noise")
    parser.add_argument(
        "--output-json", default=OUTPUT_JSON_DEFAULT,
        help="Where to save the full results grid as JSON",
    )
    args = parser.parse_args()

    if not args.all and not args.volume:
        raise SystemExit("Provide a volume path, or use --all to run the full test set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Ckpt        : {args.ckpt}")
    print(f"Corruptions : {args.corruptions}")
    print(f"Severities  : {args.severities}")

    if args.all:
        with open(TEST_LIST_DEFAULT) as fp:
            volume_paths = [osp.join(TEST_VOL_DIR_DEFAULT, line.strip()) for line in fp if line.strip()]
        print(f"Test set    : {len(volume_paths)} cases from {TEST_LIST_DEFAULT}")
    else:
        volume_paths = [args.volume]
    print()

    print("Loading model...")
    model = load_model(args.ckpt).to(device)

    # severity=0 is a no-op regardless of corruption type, so it's the same
    # run for every corruption — compute it once as "clean" and reuse.
    severities = sorted(set(args.severities))
    run_clean = 0 in severities
    nonzero_severities = [s for s in severities if s > 0]

    # runs[corruption][severity] = list of per-case results
    runs = {}

    if run_clean:
        print(f"\n--- clean (severity=0) ---")
        cases = []
        for i, vp in enumerate(volume_paths, start=1):
            print(f"[{i}/{len(volume_paths)}] {osp.basename(vp)}", end="  ")
            result = evaluate_volume(vp, model, device, corruption=None, severity=0, seed=args.seed)
            cases.append(result)
            print(f"DSC={mean_over_classes(result['per_class'], 'dsc')*100:.2f}%")
        runs["clean"] = {"0": cases}
        mean_dsc = np.mean([mean_over_classes(c["per_class"], "dsc") for c in cases])
        mean_hd = np.mean([mean_over_classes(c["per_class"], "hd95") for c in cases])
        print(f"clean severity=0  mean DSC={mean_dsc*100:.2f}%  mean HD95={mean_hd:.2f}mm  (n={len(cases)})")
        save_results(args.output_json, runs, args, severities)

    for corruption in args.corruptions:
        runs.setdefault(corruption, {})
        for severity in nonzero_severities:
            print(f"\n--- {corruption} severity={severity} ---")
            cases = []
            for i, vp in enumerate(volume_paths, start=1):
                print(f"[{i}/{len(volume_paths)}] {osp.basename(vp)}", end="  ")
                result = evaluate_volume(vp, model, device, corruption=corruption, severity=severity, seed=args.seed)
                cases.append(result)
                print(f"DSC={mean_over_classes(result['per_class'], 'dsc')*100:.2f}%")
            runs[corruption][str(severity)] = cases
            mean_dsc = np.mean([mean_over_classes(c["per_class"], "dsc") for c in cases])
            mean_hd = np.mean([mean_over_classes(c["per_class"], "hd95") for c in cases])
            print(f"{corruption} severity={severity}  mean DSC={mean_dsc*100:.2f}%  mean HD95={mean_hd:.2f}mm  (n={len(cases)})")
            save_results(args.output_json, runs, args, severities)

    # compact summary grid: rows = severity, cols = corruption (+ clean)
    print(f"\n{'='*70}\nSUMMARY (mean DSC %, all classes & cases)\n{'='*70}")
    col_names = (["clean"] if run_clean else []) + list(args.corruptions)
    header = f"{'severity':<10}" + "".join(f"{c:>16}" for c in col_names)
    print(header)
    for severity in severities:
        row = f"{severity:<10}"
        for c in col_names:
            key = "0" if c == "clean" else str(severity)
            if c == "clean" and severity != 0:
                row += f"{'':>16}"
                continue
            cases = runs.get(c, {}).get(key)
            if cases is None:
                row += f"{'':>16}"
            else:
                mean_dsc = np.mean([mean_over_classes(cc["per_class"], "dsc") for cc in cases])
                row += f"{mean_dsc*100:>15.2f}%"
        print(row)

    save_results(args.output_json, runs, args, severities)
    print(f"\nSaved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
