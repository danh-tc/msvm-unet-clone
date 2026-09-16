"""
infer_single.py — Load checkpoint, run on one 3D volume, report DSC & HD95.

Usage:
    python infer_single.py <path/to/case.npy.h5>
    python infer_single.py <path/to/case.npy.h5> --ckpt <path/to/epoch.ckpt>

Defaults:
    --ckpt  log/msvm_unet-synapse-r0/checkpoints/epoch.259-val_mean_dice.0.8500.ckpt
"""
import argparse
import json
import os
import os.path as osp
import sys
from collections import OrderedDict

import h5py
import numpy as np
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
import torch
from torchvision.transforms import transforms

# ── ensure repo root is in path ──────────────────────────────────────────────
REPO = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, REPO)

from model import build_model

# ── constants ────────────────────────────────────────────────────────────────

CKPT_DEFAULT = osp.join(
    REPO, "log", "msvm_unet-synapse-r0", "checkpoints",
    "epoch.259-val_mean_dice.0.8500.ckpt",
)
IMG_SIZE     = 224
IN_CHANNELS  = 3
NUM_CLASSES  = 9  # 1 background + 8 organs
CLASS_NAMES  = ["Aorta", "GB", "KL", "KR", "Liver", "PC", "SP", "SM"]
TEST_LIST_DEFAULT = osp.join(REPO, "lists", "lists_Synapse", "test.txt")
TEST_VOL_DIR_DEFAULT = osp.join(REPO, "data", "Synapse", "test_vol_h5")
OUTPUT_JSON_DEFAULT = osp.join(REPO, "results", "infer_single_results.json")

# Same normalization used during training (OursTransform.norm_x_transform)
NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


# ── helpers ──────────────────────────────────────────────────────────────────

def load_volume(h5_path: str):
    """Returns (image, label) as float32 numpy arrays of shape [D, H, W]."""
    with h5py.File(h5_path, "r") as f:
        image = f["image"][:]   # [D, H, W]
        label = f["label"][:]   # [D, H, W]
    return image.astype(np.float32), label.astype(np.float32)


def load_model(ckpt_path: str) -> torch.nn.Module:
    model = build_model(
        name="msvm_unet",
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("_model.", "", 1)] = v
    model.load_state_dict(state_dict)
    return model


def predict_volume(model: torch.nn.Module, volume: np.ndarray, device: str) -> np.ndarray:
    """Slice-by-slice inference. Returns prediction [D, H, W]."""
    import time
    model.eval()
    patch = (IMG_SIZE, IMG_SIZE)
    prediction = np.zeros_like(volume, dtype=np.int64)
    total = volume.shape[0]
    times = []

    for d in range(total):
        t0 = time.perf_counter()
        slc = volume[d]          # [H, W]
        h, w = slc.shape

        if h != patch[0] or w != patch[1]:
            slc_r = zoom(slc, (patch[0] / h, patch[1] / w), order=3)
        else:
            slc_r = slc

        inp = NORM(slc_r).unsqueeze(0).float().to(device)

        with torch.no_grad():
            out = model(inp)
            pred = torch.argmax(torch.softmax(out, dim=1), dim=1)
            pred = pred.squeeze(0).cpu().numpy()

        if h != patch[0] or w != patch[1]:
            pred = zoom(pred, (h / patch[0], w / patch[1]), order=0)

        prediction[d] = pred

        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        avg = sum(times) / len(times)
        remaining = avg * (total - d - 1)
        print(
            f"  Slice {d+1:>3}/{total}  |  {elapsed*1000:>6.1f}ms  |  "
            f"avg {avg*1000:>6.1f}ms  |  ETA {remaining:>5.1f}s",
            flush=True,
        )

    return prediction


def calc_dsc_hd95(pred: np.ndarray, gt: np.ndarray):
    """Binary DSC & HD95 on 3D arrays."""
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    if p.sum() > 0 and g.sum() > 0:
        dsc  = metric.binary.dc(p, g)
        hd95 = metric.binary.hd95(p, g)
    elif p.sum() == 0 and g.sum() == 0:
        dsc, hd95 = 1.0, 0.0
    else:
        dsc, hd95 = 0.0, 0.0
    return dsc, hd95


def compute_metrics(prediction: np.ndarray, label: np.ndarray) -> dict:
    """Per-class DSC/HD95 for one volume's prediction vs its ground truth."""
    per_class = {}
    for c, name in enumerate(CLASS_NAMES, start=1):
        dsc, hd95 = calc_dsc_hd95(prediction == c, label == c)
        per_class[name] = {"dsc": dsc, "hd95": hd95}
    return per_class


def print_case_table(case_name: str, per_class: dict) -> None:
    print(f"\n=== {case_name} ===")
    print(f"{'Class':<8}  {'DSC':>8}  {'HD95':>10}")
    print("-" * 32)
    dscs, hds = [], []
    for name in CLASS_NAMES:
        m = per_class[name]
        print(f"{name:<8}  {m['dsc']*100:>7.2f}%  {m['hd95']:>8.2f}mm")
        dscs.append(m["dsc"])
        hds.append(m["hd95"])
    print("-" * 32)
    print(f"{'Mean':<8}  {np.mean(dscs)*100:>7.2f}%  {np.mean(hds):>8.2f}mm")


def print_aggregate_table(case_results: list) -> None:
    """case_results: list of {"case_name": str, "per_class": dict}."""
    print(f"\n{'='*60}\nAGGREGATE over {len(case_results)} cases (mean +- std)\n{'='*60}")
    print(f"{'Class':<8}  {'DSC':>18}  {'HD95':>18}")
    print("-" * 50)
    all_dsc_means, all_hd_means = [], []
    for name in CLASS_NAMES:
        dsc_arr = np.array([r["per_class"][name]["dsc"] for r in case_results])
        hd_arr = np.array([r["per_class"][name]["hd95"] for r in case_results])
        print(
            f"{name:<8}  "
            f"{dsc_arr.mean()*100:>6.2f}% +- {dsc_arr.std()*100:>4.2f}%  "
            f"{hd_arr.mean():>6.2f} +- {hd_arr.std():>4.2f}mm"
        )
        all_dsc_means.append(dsc_arr.mean())
        all_hd_means.append(hd_arr.mean())
    print("-" * 50)
    print(f"{'Mean':<8}  DSC: {np.mean(all_dsc_means)*100:.2f}%   HD95: {np.mean(all_hd_means):.2f}mm")


def evaluate_case(volume_path: str, model: torch.nn.Module, device) -> dict:
    case_name = osp.basename(volume_path)
    image, label = load_volume(volume_path)
    print(f"Volume shape : {image.shape}  (D x H x W)")
    print(f"Label classes: {np.unique(label).astype(int).tolist()}")
    prediction = predict_volume(model, image, str(device))
    per_class = compute_metrics(prediction, label)
    return {"case_name": case_name, "per_class": per_class}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("volume", nargs="?", help="Path to a single .npy.h5 test volume")
    parser.add_argument(
        "--all", action="store_true",
        help="Run over the full test set (lists/lists_Synapse/test.txt) instead of one volume",
    )
    parser.add_argument("--ckpt", default=CKPT_DEFAULT, help="Checkpoint path")
    parser.add_argument(
        "--output-json", default=OUTPUT_JSON_DEFAULT,
        help="Where to save per-case + aggregate results as JSON (--all mode only)",
    )
    args = parser.parse_args()

    if not args.all and not args.volume:
        raise SystemExit("Provide a volume path, or use --all to run the full test set.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Ckpt   : {args.ckpt}")

    if args.all:
        with open(TEST_LIST_DEFAULT) as fp:
            volume_paths = [osp.join(TEST_VOL_DIR_DEFAULT, line.strip()) for line in fp if line.strip()]
        print(f"Test set : {len(volume_paths)} cases from {TEST_LIST_DEFAULT}")
    else:
        volume_paths = [args.volume]
    print()

    print("Loading model...")
    model = load_model(args.ckpt).to(device)

    case_results = []
    for i, volume_path in enumerate(volume_paths, start=1):
        print(f"\n[{i}/{len(volume_paths)}] {osp.basename(volume_path)}")
        result = evaluate_case(volume_path, model, device)
        case_results.append(result)
        print_case_table(result["case_name"], result["per_class"])

    if args.all:
        print_aggregate_table(case_results)
        os.makedirs(osp.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as fp:
            json.dump(
                {"config": {"ckpt": args.ckpt}, "cases": case_results},
                fp, indent=2,
            )
        print(f"\nSaved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
