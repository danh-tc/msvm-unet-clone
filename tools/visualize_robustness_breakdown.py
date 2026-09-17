#!/usr/bin/env python3
"""
visualize_robustness_breakdown.py — Show predicted masks breaking down as
corruption severity increases. Runs the frozen MSVM-UNet checkpoint on one
volume (default case0001), corrupting it at each severity exactly like
infer_robustness.py does, and renders a grid: rows = severity 0..4,
columns = [corrupted CT slice, prediction overlay, GT overlay].

Requires GPU (same selective_scan_cuda_* kernel dependency as the rest of
the pipeline — no CPU fallback).
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO)

from infer_single import CKPT_DEFAULT, CLASS_NAMES, TEST_VOL_DIR_DEFAULT, calc_dsc_hd95, load_model, load_volume, predict_volume
from corruptions import CORRUPTIONS, MAX_SEVERITY, corrupt_volume
from tools.visualize_sample import colorize_label, get_font, normalize_image


def mean_class_dsc(prediction: np.ndarray, label: np.ndarray) -> float:
    dscs = []
    for c in range(1, len(CLASS_NAMES) + 1):
        dsc, _ = calc_dsc_hd95(prediction == c, label == c)
        dscs.append(dsc)
    return float(np.mean(dscs))


def pick_slice(label: np.ndarray) -> int:
    areas = (label > 0).reshape(label.shape[0], -1).sum(axis=1)
    return int(np.argmax(areas))


def make_row(ct_slice, pred_slice, gt_slice, severity: int, dsc: float, alpha: float = 0.45) -> Image.Image:
    ct_rgb = normalize_image(ct_slice)
    pred_rgb = colorize_label(pred_slice)
    gt_rgb = colorize_label(gt_slice)

    pred_overlay = ct_rgb.copy()
    fg = pred_slice > 0
    pred_overlay[fg] = ((1 - alpha) * pred_overlay[fg] + alpha * pred_rgb[fg]).astype(np.uint8)

    gt_overlay = ct_rgb.copy()
    fg = gt_slice > 0
    gt_overlay[fg] = ((1 - alpha) * gt_overlay[fg] + alpha * gt_rgb[fg]).astype(np.uint8)

    divider = np.full((ct_rgb.shape[0], 4, 3), 255, dtype=np.uint8)
    panel = np.hstack([ct_rgb, divider, pred_overlay, divider, gt_overlay])

    label_h = 26
    out = Image.new("RGB", (panel.shape[1], panel.shape[0] + label_h), "white")
    draw = ImageDraw.Draw(out)
    font = get_font(15)
    tag = "severity=0 (clean)" if severity == 0 else f"severity={severity}"
    draw.text((8, 4), f"{tag}   volume mean DSC = {dsc*100:.2f}%", fill=(0, 0, 0), font=font)
    out.paste(Image.fromarray(panel), (0, label_h))
    return out


def make_grid(rows, corruption: str, case_name: str, z: int) -> Image.Image:
    header_h = 60
    col_w = rows[0].width
    width = col_w
    small_font = get_font(13)
    title_font = get_font(18)

    header = Image.new("RGB", (width, header_h), "white")
    draw = ImageDraw.Draw(header)
    draw.text((8, 6), f"{case_name} — {corruption} — slice {z} — prediction breakdown vs severity", fill=(0, 0, 0), font=title_font)
    col_labels_y = header_h - 20
    third = col_w // 3
    for i, name in enumerate(("Corrupted CT slice", "Prediction overlay", "GT overlay")):
        draw.text((i * third + 8, col_labels_y), name, fill=(60, 60, 60), font=small_font)

    gap = 6
    total_h = header_h + sum(r.height for r in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, total_h), "white")
    canvas.paste(header, (0, 0))
    y = header_h
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + gap
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case0001.npy.h5")
    parser.add_argument("--corruptions", nargs="+", default=["contrast_shift", "gaussian_noise"], choices=list(CORRUPTIONS))
    parser.add_argument("--severities", nargs="+", type=int, default=list(range(MAX_SEVERITY + 1)))
    parser.add_argument("--ckpt", default=CKPT_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=osp.join(REPO, "visualizations"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    volume_path = osp.join(TEST_VOL_DIR_DEFAULT, args.case)
    image, label = load_volume(volume_path)
    z = pick_slice(label)
    print(f"case={args.case}  slice={z}  device={device}")

    print("Loading model...")
    model = load_model(args.ckpt).to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    for corruption in args.corruptions:
        rows = []
        for severity in sorted(args.severities):
            if severity == 0:
                corrupted = image
            else:
                corrupted = corrupt_volume(image, corruption, severity, seed=args.seed)
            prediction = predict_volume(model, corrupted, str(device))
            dsc = mean_class_dsc(prediction, label)
            print(f"  {corruption} severity={severity}  volume mean DSC={dsc*100:.2f}%")
            rows.append(make_row(corrupted[z], prediction[z], label[z], severity, dsc))

        grid = make_grid(rows, corruption, args.case, z)
        out_path = osp.join(args.out_dir, f"breakdown_{args.case.replace('.npy.h5', '')}_{corruption}.png")
        grid.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
