"""
visualize_sam_comparison.py — GT vs Base (MSVM-UNet) vs SAM-Med2D-refined vs
MedSAM-refined contour overlays, side by side, for one case + one organ class.

Picks the N slices with the largest ground-truth area for the target class,
runs all three prediction flows on the full volume (needed since refine_volume
works slice-by-slice with neighboring-class context), and renders one row per
slice with columns [GT | Base | SAM-Med2D | MedSAM].

Usage:
    python tools/visualize_sam_comparison.py --case case0008 --class-id 4
    python tools/visualize_sam_comparison.py --case case0008 --class-id 4 --num-slices 5 \
        --out visualizations/case0008_KR_compare.png
"""
import argparse
import os
import os.path as osp
import sys

import cv2
import h5py
import numpy as np
import torch

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO)

from infer_single import CKPT_DEFAULT, CLASS_NAMES, load_model, load_volume, predict_volume
from infer_single_sam import (
    MEDSAM_CKPT_DEFAULT,
    SAM_CKPT_DEFAULT,
    build_medsam_predictor,
    build_sam_predictor,
    refine_volume,
    slice_to_rgb,
)

TEST_VOL_DIR_DEFAULT = osp.join(REPO, "data", "Synapse", "test_vol_h5")
OUT_DIR_DEFAULT = osp.join(REPO, "visualizations")

GT_COLOR = (255, 255, 0)      # yellow (RGB; image panels are RGB until the final imwrite)
BASE_COLOR = (0, 200, 0)      # green
SAMMED_COLOR = (255, 0, 0)    # red
MEDSAM_COLOR = (30, 144, 255)  # dodger blue

PANEL_H = 28  # header strip height


def dice_2d(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return 2.0 * (p & g).sum() / denom


def draw_panel(image_rgb: np.ndarray, mask: np.ndarray, color, label: str) -> np.ndarray:
    panel = image_rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, color, thickness=2)
    strip = np.zeros((PANEL_H, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(strip, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return np.vstack([strip, panel])


def resolve_case_path(case: str) -> str:
    if osp.exists(case):
        return case
    for cand in (case, case + ".npy.h5"):
        path = osp.join(TEST_VOL_DIR_DEFAULT, cand)
        if osp.exists(path):
            return path
    raise FileNotFoundError(f"Could not find case '{case}' under {TEST_VOL_DIR_DEFAULT}")


def pick_slices(label: np.ndarray, class_id: int, num_slices: int) -> list:
    areas = (label == class_id).reshape(label.shape[0], -1).sum(axis=1)
    order = np.argsort(-areas)
    top = [int(i) for i in order[:num_slices] if areas[i] > 0]
    return sorted(top)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="Case name or path, e.g. case0008 or case0008.npy.h5")
    parser.add_argument("--class-id", type=int, required=True, help="1-indexed organ class id, e.g. 4 = KR")
    parser.add_argument("--num-slices", type=int, default=3, help="Number of largest-area slices to render")
    parser.add_argument("--ckpt", default=CKPT_DEFAULT)
    parser.add_argument("--sam-checkpoint", default=SAM_CKPT_DEFAULT)
    parser.add_argument("--medsam-checkpoint", default=MEDSAM_CKPT_DEFAULT)
    parser.add_argument("--prompt", choices=["box", "box_mask"], default="box_mask")
    parser.add_argument("--dilate-iters", type=int, default=3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    volume_path = resolve_case_path(args.case)
    case_name = osp.splitext(osp.basename(volume_path))[0].replace(".npy", "")
    class_name = CLASS_NAMES[args.class_id - 1]
    out_path = args.out or osp.join(OUT_DIR_DEFAULT, f"{case_name}_{class_name}_compare.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Case  : {volume_path}")
    print(f"Class : {class_name} (id={args.class_id})")
    print(f"Device: {device}")

    image, label = load_volume(volume_path)
    slices = pick_slices(label, args.class_id, args.num_slices)
    if not slices:
        raise SystemExit(f"Class {class_name} (id={args.class_id}) has no ground-truth pixels in {volume_path}")
    print(f"Slices (largest-area first {args.num_slices}): {slices}")

    print("Loading MSVM-UNet...")
    msvm_model = load_model(args.ckpt).to(device)
    print("Loading SAM-Med2D...")
    sammed_predictor = build_sam_predictor(args.sam_checkpoint, device)
    print("Loading MedSAM...")
    medsam_predictor = build_medsam_predictor(args.medsam_checkpoint, device)

    print("Running MSVM-UNet baseline...")
    baseline_pred = predict_volume(msvm_model, image, str(device))
    print("Running SAM-Med2D refine...")
    sammed_pred = refine_volume(
        sammed_predictor, image, baseline_pred,
        use_mask_prompt=(args.prompt == "box_mask"),
        dilate_iters=args.dilate_iters, iou_threshold=args.iou_threshold,
        desc="SAM-Med2D refine",
    )
    print("Running MedSAM refine...")
    medsam_pred = refine_volume(
        medsam_predictor, image, baseline_pred,
        use_mask_prompt=False,
        dilate_iters=args.dilate_iters, iou_threshold=args.iou_threshold,
        desc="MedSAM refine",
    )

    rows = []
    for d in slices:
        image_rgb = slice_to_rgb(image[d])
        gt_mask = (label[d] == args.class_id)
        base_mask = (baseline_pred[d] == args.class_id)
        sammed_mask = (sammed_pred[d] == args.class_id)
        medsam_mask = (medsam_pred[d] == args.class_id)

        panels = [
            draw_panel(image_rgb, gt_mask, GT_COLOR, f"z={d} GT"),
            draw_panel(image_rgb, base_mask, BASE_COLOR, f"Base dsc={dice_2d(base_mask, gt_mask):.2f}"),
            draw_panel(image_rgb, sammed_mask, SAMMED_COLOR, f"SAM-Med2D dsc={dice_2d(sammed_mask, gt_mask):.2f}"),
            draw_panel(image_rgb, medsam_mask, MEDSAM_COLOR, f"MedSAM dsc={dice_2d(medsam_mask, gt_mask):.2f}"),
        ]
        rows.append(np.hstack(panels))

    grid = np.vstack(rows)
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
