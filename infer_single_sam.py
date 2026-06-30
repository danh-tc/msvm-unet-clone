"""
infer_single_sam.py — MSVM-UNet baseline vs MSVM-UNet + SAM-Med2D refine.

For each slice, takes MSVM-UNet's predicted mask per organ class, derives a
box + low-res mask prompt from it, and asks SAM-Med2D to re-segment that
organ. Reports DSC & HD95 for both flows side by side.

Usage:
    # single volume
    python infer_single_sam.py <path/to/case.npy.h5>

    # full test set (lists/lists_Synapse/test.txt), aggregated mean +- std
    python infer_single_sam.py --all

    python infer_single_sam.py --all \
        --ckpt <path/to/msvm_epoch.ckpt> \
        --sam-checkpoint <path/to/sam-med2d_b.pth> \
        --prompt box_mask \
        --output-json results/infer_single_sam_results.json

Defaults:
    --ckpt           log/msvm_unet-synapse-r0/checkpoints/epoch.259-val_mean_dice.0.8500.ckpt
    --sam-checkpoint SAM-Med2D/pretrain_model/sam-med2d_b.pth
    --prompt         box_mask   (box + dense mask prompt; "box" for box-only)
    --output-json    results/infer_single_sam_results.json   (--all mode only)
"""
import argparse
import json
import os
import os.path as osp
import sys
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import torch
from medpy import metric
from tqdm import tqdm

# ── ensure repo root + SAM-Med2D are in path ─────────────────────────────────
REPO = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, osp.join(REPO, "SAM-Med2D"))

from infer_single import (
    CKPT_DEFAULT,
    CLASS_NAMES,
    calc_dsc_hd95,
    load_model,
    load_volume,
    predict_volume,
)
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor

SAM_CKPT_DEFAULT = osp.join(REPO, "SAM-Med2D", "pretrain_model", "sam-med2d_b.pth")
SAM_IMAGE_SIZE = 256
MIN_AREA = 20  # skip classes with fewer predicted pixels on a slice (noise, degenerate box)
TEST_LIST_DEFAULT = osp.join(REPO, "lists", "lists_Synapse", "test.txt")
TEST_VOL_DIR_DEFAULT = osp.join(REPO, "data", "Synapse", "test_vol_h5")
OUTPUT_JSON_DEFAULT = osp.join(REPO, "results", "infer_single_sam_results.json")


# ── helpers ──────────────────────────────────────────────────────────────────

def build_sam_predictor(checkpoint: str, device: torch.device) -> SammedPredictor:
    model_args = SimpleNamespace(
        image_size=SAM_IMAGE_SIZE,
        encoder_adapter=True,
        sam_checkpoint=checkpoint,
    )
    model = sam_model_registry["vit_b"](model_args).to(device)
    model.eval()
    return SammedPredictor(model)


def slice_to_rgb(slc: np.ndarray) -> np.ndarray:
    """Grayscale [0,1] float slice -> uint8 RGB for SAM-Med2D."""
    u8 = (np.clip(slc, 0.0, 1.0) * 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def mask_prompt_input(mask: np.ndarray) -> np.ndarray:
    """Binary mask at original resolution -> [1, 64, 64] low-res logits prompt."""
    m64 = cv2.resize(mask.astype(np.float32), (64, 64), interpolation=cv2.INTER_LINEAR)
    logits = (m64 - 0.5) * 20.0
    return logits[None, :, :].astype(np.float32)


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    union = (a | b).sum()
    if union == 0:
        return 1.0
    return (a & b).sum() / union


FALLBACK_SCORE = 1.0  # baseline-fallback masks outrank any SAM score (range ~0-1) in overlap compositing


def refine_volume_with_sam(
    sam_predictor: SammedPredictor,
    image: np.ndarray,
    baseline_pred: np.ndarray,
    use_mask_prompt: bool,
    dilate_iters: int,
    iou_threshold: float,
) -> np.ndarray:
    """Per-slice, per-class SAM-Med2D refinement of an MSVM-UNet prediction.

    Each class's SAM mask is clipped to a dilated band around its own baseline
    mask (so it can only refine boundaries, not invade neighboring organs),
    and is discarded in favor of the baseline mask if it diverges too much
    (IoU vs baseline < iou_threshold) — guards against catastrophic SAM errors.
    """
    refined = np.zeros_like(baseline_pred)
    total = image.shape[0]
    kernel = np.ones((3, 3), np.uint8)

    for d in tqdm(range(total), desc="SAM refine", unit="slice"):
        classes = [c for c in np.unique(baseline_pred[d]) if c > 0]
        if not classes:
            continue

        image_rgb = slice_to_rgb(image[d])
        sam_predictor.set_image(image_rgb, image_format="RGB")

        score_map = np.full(baseline_pred[d].shape, -np.inf, dtype=np.float32)

        for c in classes:
            mask_c = (baseline_pred[d] == c)
            if mask_c.sum() < MIN_AREA:
                continue

            ys, xs = np.where(mask_c)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            mask_input = mask_prompt_input(mask_c) if use_mask_prompt else None

            masks, scores, _ = sam_predictor.predict(
                box=box, mask_input=mask_input, multimask_output=True,
            )
            best = int(np.argmax(scores))
            sam_mask, sam_score = masks[best].astype(bool), float(scores[best])

            # restrict SAM's mask to a dilated band around its own baseline mask
            dilated = cv2.dilate(mask_c.astype(np.uint8), kernel, iterations=dilate_iters).astype(bool)
            sam_mask_clipped = sam_mask & dilated

            if binary_iou(sam_mask_clipped, mask_c) >= iou_threshold:
                final_mask, final_score = sam_mask_clipped, sam_score
            else:
                final_mask, final_score = mask_c, FALLBACK_SCORE

            update = final_mask & (final_score > score_map)
            refined[d][update] = c
            score_map[update] = final_score

    return refined


def compute_metrics(baseline: np.ndarray, refined: np.ndarray, label: np.ndarray) -> dict:
    """Per-class DSC/HD95 for baseline vs SAM-refined, keyed by class name."""
    per_class = {}
    for c, name in enumerate(CLASS_NAMES, start=1):
        b_dsc, b_hd = calc_dsc_hd95(baseline == c, label == c)
        s_dsc, s_hd = calc_dsc_hd95(refined == c, label == c)
        per_class[name] = {
            "base_dsc": b_dsc, "sam_dsc": s_dsc,
            "base_hd95": b_hd, "sam_hd95": s_hd,
        }
    return per_class


def print_case_table(case_name: str, per_class: dict) -> None:
    print(f"\n=== {case_name} ===")
    print(f"{'Class':<8}  {'Base DSC':>9}  {'SAM DSC':>9}  {'Delta':>7}  "
          f"{'Base HD95':>10}  {'SAM HD95':>10}")
    print("-" * 64)
    base_dscs, sam_dscs, base_hds, sam_hds = [], [], [], []
    for name in CLASS_NAMES:
        m = per_class[name]
        delta = (m["sam_dsc"] - m["base_dsc"]) * 100
        print(
            f"{name:<8}  {m['base_dsc']*100:>8.2f}%  {m['sam_dsc']*100:>8.2f}%  {delta:>+6.2f}%  "
            f"{m['base_hd95']:>9.2f}mm  {m['sam_hd95']:>9.2f}mm"
        )
        base_dscs.append(m["base_dsc"]); sam_dscs.append(m["sam_dsc"])
        base_hds.append(m["base_hd95"]); sam_hds.append(m["sam_hd95"])
    print("-" * 64)
    mean_delta = (np.mean(sam_dscs) - np.mean(base_dscs)) * 100
    print(
        f"{'Mean':<8}  {np.mean(base_dscs)*100:>8.2f}%  {np.mean(sam_dscs)*100:>8.2f}%  "
        f"{mean_delta:>+6.2f}%  {np.mean(base_hds):>9.2f}mm  {np.mean(sam_hds):>9.2f}mm"
    )


def print_aggregate_table(case_results: list) -> None:
    """case_results: list of {"case_name": str, "per_class": dict}."""
    print(f"\n{'='*80}\nAGGREGATE over {len(case_results)} cases (mean +- std)\n{'='*80}")
    print(f"{'Class':<8}  {'Base DSC':>15}  {'SAM DSC':>15}  {'Delta':>7}  "
          f"{'Base HD95':>14}  {'SAM HD95':>14}")
    print("-" * 90)

    all_base_dscs, all_sam_dscs = [], []
    for name in CLASS_NAMES:
        b_dscs = np.array([r["per_class"][name]["base_dsc"] for r in case_results])
        s_dscs = np.array([r["per_class"][name]["sam_dsc"] for r in case_results])
        b_hds = np.array([r["per_class"][name]["base_hd95"] for r in case_results])
        s_hds = np.array([r["per_class"][name]["sam_hd95"] for r in case_results])
        delta = (s_dscs.mean() - b_dscs.mean()) * 100
        print(
            f"{name:<8}  {b_dscs.mean()*100:>6.2f}% +- {b_dscs.std()*100:>4.2f}%  "
            f"{s_dscs.mean()*100:>6.2f}% +- {s_dscs.std()*100:>4.2f}%  {delta:>+6.2f}%  "
            f"{b_hds.mean():>6.2f} +- {b_hds.std():>4.2f}mm  {s_hds.mean():>6.2f} +- {s_hds.std():>4.2f}mm"
        )
        all_base_dscs.append(b_dscs.mean()); all_sam_dscs.append(s_dscs.mean())

    print("-" * 90)
    mean_delta = (np.mean(all_sam_dscs) - np.mean(all_base_dscs)) * 100
    print(f"{'Mean':<8}  Base DSC: {np.mean(all_base_dscs)*100:.2f}%   "
          f"SAM DSC: {np.mean(all_sam_dscs)*100:.2f}%   Delta: {mean_delta:+.2f}%")


def evaluate_case(
    volume_path: str,
    msvm_model: torch.nn.Module,
    sam_predictor: SammedPredictor,
    device,
    use_mask_prompt: bool,
    dilate_iters: int,
    iou_threshold: float,
) -> dict:
    case_name = osp.basename(volume_path)
    image, label = load_volume(volume_path)
    baseline_pred = predict_volume(msvm_model, image, str(device))
    refined_pred = refine_volume_with_sam(
        sam_predictor, image, baseline_pred,
        use_mask_prompt=use_mask_prompt,
        dilate_iters=dilate_iters,
        iou_threshold=iou_threshold,
    )
    per_class = compute_metrics(baseline_pred, refined_pred, label)
    return {"case_name": case_name, "per_class": per_class}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("volume", nargs="?", help="Path to a single .npy.h5 test volume")
    parser.add_argument(
        "--all", action="store_true",
        help="Run over the full test set (lists/lists_Synapse/test.txt) instead of one volume",
    )
    parser.add_argument("--ckpt", default=CKPT_DEFAULT, help="MSVM-UNet checkpoint path")
    parser.add_argument("--sam-checkpoint", default=SAM_CKPT_DEFAULT, help="SAM-Med2D checkpoint path")
    parser.add_argument(
        "--prompt", choices=["box", "box_mask"], default="box_mask",
        help="SAM prompt type: box-only, or box + dense mask prompt (recommended, more accurate)",
    )
    parser.add_argument(
        "--dilate-iters", type=int, default=3,
        help="Dilate baseline mask by N 3x3-kernel iterations to bound SAM's refine region (0 = unbounded)",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="Discard SAM mask and fall back to baseline if IoU(sam, baseline) < this",
    )
    parser.add_argument(
        "--output-json", default=OUTPUT_JSON_DEFAULT,
        help="Where to save per-case + aggregate results as JSON (--all mode only)",
    )
    args = parser.parse_args()

    if not args.all and not args.volume:
        raise SystemExit("Provide a volume path, or use --all to run the full test set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"MSVM ckpt: {args.ckpt}")
    print(f"SAM ckpt : {args.sam_checkpoint}")
    print(f"Prompt   : {args.prompt}")
    print(f"Dilate iters  : {args.dilate_iters}")
    print(f"IoU threshold : {args.iou_threshold}")

    if args.all:
        with open(TEST_LIST_DEFAULT) as fp:
            volume_paths = [osp.join(TEST_VOL_DIR_DEFAULT, line.strip()) for line in fp if line.strip()]
        print(f"Test set : {len(volume_paths)} cases from {TEST_LIST_DEFAULT}")
    else:
        volume_paths = [args.volume]
    print()

    # load both models once, reuse across all cases
    print("Loading MSVM-UNet...")
    msvm_model = load_model(args.ckpt).to(device)
    print("Loading SAM-Med2D...")
    sam_predictor = build_sam_predictor(args.sam_checkpoint, device)

    case_results = []
    for i, volume_path in enumerate(volume_paths, start=1):
        print(f"\n[{i}/{len(volume_paths)}] {osp.basename(volume_path)}")
        result = evaluate_case(
            volume_path, msvm_model, sam_predictor, device,
            use_mask_prompt=(args.prompt == "box_mask"),
            dilate_iters=args.dilate_iters,
            iou_threshold=args.iou_threshold,
        )
        case_results.append(result)
        print_case_table(result["case_name"], result["per_class"])

    if args.all:
        print_aggregate_table(case_results)
        os.makedirs(osp.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as fp:
            json.dump(
                {
                    "config": {
                        "ckpt": args.ckpt, "sam_checkpoint": args.sam_checkpoint,
                        "prompt": args.prompt, "dilate_iters": args.dilate_iters,
                        "iou_threshold": args.iou_threshold,
                    },
                    "cases": case_results,
                },
                fp, indent=2,
            )
        print(f"\nSaved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
