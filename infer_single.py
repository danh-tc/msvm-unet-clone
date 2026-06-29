"""
infer_single.py — Load checkpoint, run on one 3D volume, report DSC & HD95.

Usage:
    python infer_single.py <path/to/case.npy.h5>
    python infer_single.py <path/to/case.npy.h5> --ckpt <path/to/epoch.ckpt>

Defaults:
    --ckpt  log/msvm_unet-synapse-r0/checkpoints/epoch.259-val_mean_dice.0.8500.ckpt
"""
import argparse
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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("volume", help="Path to .npy.h5 test volume")
    parser.add_argument("--ckpt", default=CKPT_DEFAULT, help="Checkpoint path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Volume : {args.volume}")
    print(f"Ckpt   : {args.ckpt}")
    print()

    # 1. load 3D volume
    image, label = load_volume(args.volume)
    print(f"Volume shape : {image.shape}  (D x H x W)")
    print(f"Label classes: {np.unique(label).astype(int).tolist()}")
    print()

    # 2. load model
    print("Loading model...")
    model = load_model(args.ckpt).to(device)

    # 3. slice-by-slice inference (3D → 2D → aggregate back to 3D)
    print("Running inference...")
    prediction = predict_volume(model, image, device)

    # 4. DSC & HD95 per class
    import time
    print(f"\n{'Class':<8}  {'DSC':>8}  {'HD95':>10}  {'Time':>7}")
    print("-" * 42)
    dscs, hds = [], []
    for c, name in enumerate(CLASS_NAMES, start=1):
        t0 = time.perf_counter()
        print(f"  Computing {name}...", end="\r", flush=True)
        dsc, hd95 = calc_dsc_hd95(prediction == c, label == c)
        elapsed = time.perf_counter() - t0
        print(f"{name:<8}  {dsc*100:>7.2f}%  {hd95:>8.2f}mm  {elapsed:>6.2f}s")
        dscs.append(dsc)
        hds.append(hd95)

    print("-" * 42)
    print(f"{'Mean':<8}  {np.mean(dscs)*100:>7.2f}%  {np.mean(hds):>8.2f}mm")


if __name__ == "__main__":
    main()
