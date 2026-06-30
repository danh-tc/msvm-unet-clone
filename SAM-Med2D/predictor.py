import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal SAM-Med2D predictor script.")
    parser.add_argument("--checkpoint", required=True, help="Path to sam-med2d_b.pth.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output-dir", default="outputs", help="Where to save mask and overlay.")
    parser.add_argument("--point", nargs=2, type=int, metavar=("X", "Y"), help="Positive point prompt.")
    parser.add_argument("--box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), help="Box prompt.")
    parser.add_argument("--gt-mask", help="Optional ground-truth mask path for side-by-side comparison.")
    parser.add_argument("--image-size", type=int, default=256, help="Model input size used by SAM-Med2D.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return parser.parse_args()


def build_predictor(checkpoint, image_size, device):
    model_args = SimpleNamespace(
        image_size=image_size,
        encoder_adapter=True,
        sam_checkpoint=checkpoint,
    )
    model = sam_model_registry["vit_b"](model_args).to(device)
    model.eval()
    return SammedPredictor(model)


def color_overlay(image_bgr, mask, color):
    overlay = image_bgr.copy()
    overlay[mask.astype(bool)] = color
    return cv2.addWeighted(image_bgr, 0.65, overlay, 0.35, 0)


def mask_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    iou = intersection / union if union else 1.0
    dice = (2 * intersection) / (pred_sum + gt_sum) if pred_sum + gt_sum else 1.0
    return iou, dice


def save_outputs(image_bgr, mask, output_dir, gt_mask_path=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_path = output_dir / "mask.png"
    cv2.imwrite(str(mask_path), mask_u8)

    overlay = color_overlay(image_bgr, mask, (0, 255, 0))
    overlay_path = output_dir / "overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    outputs = {"mask": mask_path, "overlay": overlay_path}

    if gt_mask_path:
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise SystemExit(f"Could not read GT mask: {gt_mask_path}")
        gt_mask = gt_mask > 0
        if gt_mask.shape != mask.shape:
            gt_mask = cv2.resize(
                gt_mask.astype(np.uint8),
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        gt_overlay = color_overlay(image_bgr, gt_mask, (0, 0, 255))
        gt_overlay_path = output_dir / "gt_overlay.png"
        cv2.imwrite(str(gt_overlay_path), gt_overlay)

        comparison = np.concatenate([image_bgr, overlay, gt_overlay], axis=1)
        comparison_path = output_dir / "comparison.png"
        cv2.imwrite(str(comparison_path), comparison)

        iou, dice = mask_metrics(mask, gt_mask)
        outputs.update(
            {
                "gt_overlay": gt_overlay_path,
                "comparison": comparison_path,
                "iou": iou,
                "dice": dice,
            }
        )

    return outputs


def main():
    args = parse_args()
    if not args.point and not args.box:
        raise SystemExit("Provide either --point X Y or --box X1 Y1 X2 Y2.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    predictor = build_predictor(args.checkpoint, args.image_size, device)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb, image_format="RGB")

    if args.point:
        point_coords = np.array([args.point])
        point_labels = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
    else:
        box = np.array(args.box)
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True,
        )

    best_idx = int(np.argmax(scores))
    mask = masks[best_idx]
    outputs = save_outputs(image_bgr, mask, args.output_dir, args.gt_mask)

    print(f"device: {device}")
    print(f"score: {float(scores[best_idx]):.4f}")
    print(f"mask: {outputs['mask']}")
    print(f"overlay: {outputs['overlay']}")
    if args.gt_mask:
        print(f"gt_overlay: {outputs['gt_overlay']}")
        print(f"comparison: {outputs['comparison']}")
        print(f"iou: {outputs['iou']:.4f}")
        print(f"dice: {outputs['dice']:.4f}")


if __name__ == "__main__":
    main()
