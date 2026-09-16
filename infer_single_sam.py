"""
infer_single_sam.py — MSVM-UNet baseline vs +SAM-Med2D refine vs +MedSAM refine.

For each slice, takes MSVM-UNet's predicted mask per organ class, derives a
box (+ low-res mask, for SAM-Med2D) prompt from it, and asks SAM-Med2D and
MedSAM to each independently re-segment that organ. Reports DSC & HD95 for
all three flows side by side.

Usage:
    # single volume
    python infer_single_sam.py <path/to/case.npy.h5>

    # full test set (lists/lists_Synapse/test.txt), aggregated mean +- std
    python infer_single_sam.py --all

    python infer_single_sam.py --all \
        --ckpt <path/to/msvm_epoch.ckpt> \
        --sam-checkpoint <path/to/sam-med2d_b.pth> \
        --medsam-checkpoint <path/to/medsam_vit_b.pth> \
        --prompt box_mask \
        --output-json results/infer_single_sam_results.json

Defaults:
    --ckpt              log/msvm_unet-synapse-r0/checkpoints/epoch.259-val_mean_dice.0.8500.ckpt
    --sam-checkpoint    SAM-Med2D/pretrain_model/sam-med2d_b.pth
    --medsam-checkpoint MedSAM/work_dir/MedSAM/medsam_vit_b.pth
    --prompt            box_mask   (box + dense mask prompt for SAM-Med2D; "box" for box-only.
                         MedSAM is always box-only, matching its released inference recipe.)
    --output-json       results/infer_single_sam_results.json   (--all mode only)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import os.path as osp
import sys
import types
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from skimage import transform as sk_transform
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
MEDSAM_CKPT_DEFAULT = osp.join(REPO, "MedSAM", "work_dir", "MedSAM", "medsam_vit_b.pth")
MEDSAM_IMAGE_SIZE = 1024
MIN_AREA = 20  # skip classes with fewer predicted pixels on a slice (noise, degenerate box)
TEST_LIST_DEFAULT = osp.join(REPO, "lists", "lists_Synapse", "test.txt")
TEST_VOL_DIR_DEFAULT = osp.join(REPO, "data", "Synapse", "test_vol_h5")
OUTPUT_JSON_DEFAULT = osp.join(REPO, "results", "infer_single_sam_results.json")


def _load_medsam_sam_registry():
    """Load MedSAM's `sam_model_registry` without adding MedSAM/ to sys.path.

    MedSAM ships its own `segment_anything` package, whose top-level name
    collides with SAM-Med2D's (already imported above as the real
    `segment_anything` module). We load MedSAM's package tree under a private
    alias instead, and only pull in `build_sam` + `modeling` — `predictor.py`
    and `automatic_mask_generator.py` are skipped because they hardcode an
    absolute `from segment_anything...` import that would otherwise resolve
    to SAM-Med2D's package (already in sys.modules) and break.
    """
    alias = "_medsam_segment_anything"
    root = osp.join(REPO, "MedSAM", "segment_anything")

    pkg = types.ModuleType(alias)
    pkg.__path__ = [root]
    sys.modules[alias] = pkg

    spec = importlib.util.spec_from_file_location(
        f"{alias}.build_sam", osp.join(root, "build_sam.py"),
    )
    build_sam_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build_sam_mod
    spec.loader.exec_module(build_sam_mod)
    return build_sam_mod.sam_model_registry


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


class MedSAMPredictor:
    """Box-prompt-only predictor for a MedSAM-finetuned vanilla-SAM (ViT-B) checkpoint.

    Mirrors MedSAM/MedSAM_Inference.py's pipeline (resize to 1024x1024, box
    prompt, sigmoid + threshold 0.5), adapted to the same set_image()/predict()
    interface SammedPredictor exposes so both can share one refine loop.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._embedding = None
        self._h = None
        self._w = None

    @torch.no_grad()
    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        h, w = image.shape[:2]
        img_1024 = sk_transform.resize(
            image, (MEDSAM_IMAGE_SIZE, MEDSAM_IMAGE_SIZE), order=3,
            preserve_range=True, anti_aliasing=True,
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        self._embedding = self.model.image_encoder(img_tensor)  # (1, 256, 64, 64)
        self._h, self._w = h, w

    @torch.no_grad()
    def predict(
        self, box: np.ndarray, point_coords=None, point_labels=None,
        mask_input=None, multimask_output: bool = False,
    ):
        """Box-only prompt; point_coords/point_labels/mask_input/multimask_output are
        accepted for interface parity with SammedPredictor but unused — MedSAM's
        release recipe is box-only, single-mask."""
        box_1024 = box.astype(np.float32) / np.array(
            [self._w, self._h, self._w, self._h], dtype=np.float32
        ) * MEDSAM_IMAGE_SIZE
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=self.device)[None, None, :]

        sparse_emb, dense_emb = self.model.prompt_encoder(points=None, boxes=box_torch, masks=None)
        low_res_logits, iou_pred = self.model.mask_decoder(
            image_embeddings=self._embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        pred = torch.sigmoid(low_res_logits)
        pred = F.interpolate(pred, size=(self._h, self._w), mode="bilinear", align_corners=False)
        mask = (pred.squeeze(0).squeeze(0).cpu().numpy() > 0.5)
        score = float(iou_pred.squeeze().cpu().numpy())
        return mask[None, :, :], np.array([score], dtype=np.float32), low_res_logits


def build_medsam_predictor(checkpoint: str, device: torch.device) -> MedSAMPredictor:
    medsam_registry = _load_medsam_sam_registry()
    model = medsam_registry["vit_b"](checkpoint=checkpoint).to(device)
    model.eval()
    return MedSAMPredictor(model, device)


def slice_to_rgb(slc: np.ndarray) -> np.ndarray:
    """Grayscale [0,1] float slice -> uint8 RGB for SAM-Med2D."""
    u8 = (np.clip(slc, 0.0, 1.0) * 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def mask_prompt_input(mask: np.ndarray) -> np.ndarray:
    """Binary mask at original resolution -> [1, 64, 64] low-res logits prompt."""
    m64 = cv2.resize(mask.astype(np.float32), (64, 64), interpolation=cv2.INTER_LINEAR)
    logits = (m64 - 0.5) * 20.0
    return logits[None, :, :].astype(np.float32)


def extract_points(mask: np.ndarray, dilated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One positive + one negative point prompt from a binary mask, via distance
    transform (no FastGeodis dependency needed — cv2 is enough):

    - Positive point: the mask's most "interior" pixel (max distance to the
      mask's own boundary) — robust to noisy/ragged boundaries, unlike a
      boundary or centroid pick that can land outside a concave mask.
    - Negative point: the pixel in the dilated refine band (but outside the
      mask) farthest from the mask — anchors right at the edge of the allowed
      refine region, reinforcing the same anti-leak intent as the dilate clip.
    """
    mask_u8 = mask.astype(np.uint8)
    dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    py, px = np.unravel_index(np.argmax(dist_in), dist_in.shape)
    coords = [[px, py]]
    labels = [1]

    band = dilated & ~mask
    if band.any():
        dist_out = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        dist_out[~band] = -1
        ny, nx = np.unravel_index(np.argmax(dist_out), dist_out.shape)
        coords.append([nx, ny])
        labels.append(0)

    return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int32)


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    union = (a | b).sum()
    if union == 0:
        return 1.0
    return (a & b).sum() / union


FALLBACK_SCORE = 1.0  # baseline-fallback masks outrank any SAM score (range ~0-1) in overlap compositing


def refine_volume(
    predictor,
    image: np.ndarray,
    baseline_pred: np.ndarray,
    use_mask_prompt: bool,
    dilate_iters: int,
    iou_threshold: float,
    desc: str = "SAM refine",
    use_points: bool = False,
) -> np.ndarray:
    """Per-slice, per-class refinement of an MSVM-UNet prediction via any
    predictor exposing SammedPredictor's set_image()/predict() interface
    (SAM-Med2D's SammedPredictor or MedSAMPredictor).

    Each class's predicted mask is clipped to a dilated band around its own
    baseline mask (so it can only refine boundaries, not invade neighboring
    organs), and is discarded in favor of the baseline mask if it diverges too
    much (IoU vs baseline < iou_threshold) — guards against catastrophic
    refiner errors.
    """
    refined = np.zeros_like(baseline_pred)
    total = image.shape[0]
    kernel = np.ones((3, 3), np.uint8)

    for d in tqdm(range(total), desc=desc, unit="slice"):
        classes = [c for c in np.unique(baseline_pred[d]) if c > 0]
        if not classes:
            continue

        image_rgb = slice_to_rgb(image[d])
        predictor.set_image(image_rgb, image_format="RGB")

        score_map = np.full(baseline_pred[d].shape, -np.inf, dtype=np.float32)

        for c in classes:
            mask_c = (baseline_pred[d] == c)
            if mask_c.sum() < MIN_AREA:
                continue

            ys, xs = np.where(mask_c)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            mask_input = mask_prompt_input(mask_c) if use_mask_prompt else None

            # restrict the refiner's mask to a dilated band around its own baseline mask
            dilated = cv2.dilate(mask_c.astype(np.uint8), kernel, iterations=dilate_iters).astype(bool)

            if use_points:
                point_coords, point_labels = extract_points(mask_c, dilated)
            else:
                point_coords, point_labels = None, None

            masks, scores, _ = predictor.predict(
                point_coords=point_coords, point_labels=point_labels,
                box=box, mask_input=mask_input, multimask_output=True,
            )
            best = int(np.argmax(scores))
            pred_mask, pred_score = masks[best].astype(bool), float(scores[best])

            pred_mask_clipped = pred_mask & dilated

            if binary_iou(pred_mask_clipped, mask_c) >= iou_threshold:
                final_mask, final_score = pred_mask_clipped, pred_score
            else:
                final_mask, final_score = mask_c, FALLBACK_SCORE

            update = final_mask & (final_score > score_map)
            refined[d][update] = c
            score_map[update] = final_score

    return refined


def compute_metrics(
    baseline: np.ndarray, sammed: np.ndarray, medsam: np.ndarray | None, label: np.ndarray,
) -> dict:
    """Per-class DSC/HD95 for baseline vs SAM-Med2D-refined vs (optionally) MedSAM-refined,
    keyed by class name. medsam=None (--skip-medsam) omits the medsam_* keys entirely."""
    per_class = {}
    for c, name in enumerate(CLASS_NAMES, start=1):
        b_dsc, b_hd = calc_dsc_hd95(baseline == c, label == c)
        s_dsc, s_hd = calc_dsc_hd95(sammed == c, label == c)
        entry = {"base_dsc": b_dsc, "sammed_dsc": s_dsc, "base_hd95": b_hd, "sammed_hd95": s_hd}
        if medsam is not None:
            m_dsc, m_hd = calc_dsc_hd95(medsam == c, label == c)
            entry["medsam_dsc"] = m_dsc
            entry["medsam_hd95"] = m_hd
        per_class[name] = entry
    return per_class


# (key, column label, is_percent) — medsam_* columns are dropped automatically when absent
# from the per_class dict (i.e. when running with --skip-medsam).
_METRIC_COLUMNS = [
    ("base_dsc", "Base DSC", True),
    ("sammed_dsc", "SAMMed2D DSC", True),
    ("medsam_dsc", "MedSAM DSC", True),
    ("base_hd95", "Base HD95", False),
    ("sammed_hd95", "SAMMed2D HD95", False),
    ("medsam_hd95", "MedSAM HD95", False),
]


def _active_columns(per_class: dict) -> list:
    has_medsam = any("medsam_dsc" in m for m in per_class.values())
    return [col for col in _METRIC_COLUMNS if has_medsam or "medsam" not in col[0]]


def _fmt(value: float, is_percent: bool) -> str:
    return f"{value*100:>11.2f}%" if is_percent else f"{value:>11.2f}mm"


def print_case_table(case_name: str, per_class: dict) -> None:
    print(f"\n=== {case_name} ===")
    cols = _active_columns(per_class)
    header = f"{'Class':<8}  " + "  ".join(f"{label:>13}" for _, label, _ in cols)
    print(header)
    print("-" * len(header))
    sums = {key: [] for key, _, _ in cols}
    for name in CLASS_NAMES:
        m = per_class[name]
        print(f"{name:<8}  " + "  ".join(_fmt(m[key], pct) for key, _, pct in cols))
        for key, _, _ in cols:
            sums[key].append(m[key])
    print("-" * len(header))
    print(f"{'Mean':<8}  " + "  ".join(_fmt(np.mean(sums[key]), pct) for key, _, pct in cols))


def print_aggregate_table(case_results: list) -> None:
    """case_results: list of {"case_name": str, "per_class": dict}."""
    print(f"\n{'='*110}\nAGGREGATE over {len(case_results)} cases (mean +- std)\n{'='*110}")
    cols = _active_columns(case_results[0]["per_class"])
    header = f"{'Class':<8}  " + "  ".join(f"{label:>24}" for _, label, _ in cols)
    print(header)
    print("-" * len(header))

    all_means = {key: [] for key, _, _ in cols}
    for name in CLASS_NAMES:
        row = []
        for key, _, pct in cols:
            arr = np.array([r["per_class"][name][key] for r in case_results])
            row.append(
                f"{arr.mean()*100:>6.2f}% +- {arr.std()*100:>4.2f}%" if pct
                else f"{arr.mean():>6.2f} +- {arr.std():>4.2f}mm"
            )
            all_means[key].append(arr.mean())
        print(f"{name:<8}  " + "  ".join(f"{v:>24}" for v in row))

    print("-" * len(header))
    summary = "   ".join(
        f"{label}: {np.mean(all_means[key])*100:.2f}%" if pct
        else f"{label}: {np.mean(all_means[key]):.2f}mm"
        for key, label, pct in cols
    )
    print(f"{'Mean':<8}  {summary}")


def evaluate_case(
    volume_path: str,
    msvm_model: torch.nn.Module,
    sammed_predictor: SammedPredictor,
    medsam_predictor: MedSAMPredictor | None,
    device,
    use_mask_prompt: bool,
    dilate_iters: int,
    iou_threshold: float,
    use_points: bool = False,
) -> dict:
    """medsam_predictor=None (--skip-medsam) skips the MedSAM refine pass and its metrics.
    use_points only applies to the SAM-Med2D pass — MedSAM stays box-only, matching its
    released inference recipe."""
    case_name = osp.basename(volume_path)
    image, label = load_volume(volume_path)
    baseline_pred = predict_volume(msvm_model, image, str(device))
    sammed_pred = refine_volume(
        sammed_predictor, image, baseline_pred,
        use_mask_prompt=use_mask_prompt,
        dilate_iters=dilate_iters,
        iou_threshold=iou_threshold,
        desc="SAM-Med2D refine",
        use_points=use_points,
    )
    if medsam_predictor is not None:
        medsam_pred = refine_volume(
            medsam_predictor, image, baseline_pred,
            use_mask_prompt=False,  # MedSAM's release recipe is box-only
            dilate_iters=dilate_iters,
            iou_threshold=iou_threshold,
            desc="MedSAM refine",
        )
    else:
        medsam_pred = None
    per_class = compute_metrics(baseline_pred, sammed_pred, medsam_pred, label)
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
    parser.add_argument("--medsam-checkpoint", default=MEDSAM_CKPT_DEFAULT, help="MedSAM (vit_b) checkpoint path")
    parser.add_argument(
        "--skip-medsam", action="store_true",
        help="Skip loading/running MedSAM entirely (e.g. checkpoint not downloaded yet). "
             "Reports baseline vs SAM-Med2D-refine only.",
    )
    parser.add_argument(
        "--prompt", choices=["box", "box_mask"], default="box_mask",
        help="SAM-Med2D prompt type: box-only, or box + dense mask prompt (recommended, more accurate). "
             "MedSAM is always box-only, matching its released inference recipe.",
    )
    parser.add_argument(
        "--use-points", action="store_true",
        help="Add one positive + one negative point prompt (distance-transform-based) to "
             "the SAM-Med2D pass, on top of --prompt. MedSAM stays box-only regardless.",
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
    print(f"MSVM ckpt   : {args.ckpt}")
    print(f"SAM ckpt    : {args.sam_checkpoint}")
    print(f"MedSAM ckpt : {'skipped (--skip-medsam)' if args.skip_medsam else args.medsam_checkpoint}")
    print(f"Prompt   : {args.prompt}")
    print(f"Use points    : {args.use_points}")
    print(f"Dilate iters  : {args.dilate_iters}")
    print(f"IoU threshold : {args.iou_threshold}")

    if args.all:
        with open(TEST_LIST_DEFAULT) as fp:
            volume_paths = [osp.join(TEST_VOL_DIR_DEFAULT, line.strip()) for line in fp if line.strip()]
        print(f"Test set : {len(volume_paths)} cases from {TEST_LIST_DEFAULT}")
    else:
        volume_paths = [args.volume]
    print()

    # load all three models once, reuse across all cases
    print("Loading MSVM-UNet...")
    msvm_model = load_model(args.ckpt).to(device)
    print("Loading SAM-Med2D...")
    sammed_predictor = build_sam_predictor(args.sam_checkpoint, device)
    if args.skip_medsam:
        print("Skipping MedSAM (--skip-medsam)")
        medsam_predictor = None
    else:
        print("Loading MedSAM...")
        medsam_predictor = build_medsam_predictor(args.medsam_checkpoint, device)

    case_results = []
    for i, volume_path in enumerate(volume_paths, start=1):
        print(f"\n[{i}/{len(volume_paths)}] {osp.basename(volume_path)}")
        result = evaluate_case(
            volume_path, msvm_model, sammed_predictor, medsam_predictor, device,
            use_mask_prompt=(args.prompt == "box_mask"),
            dilate_iters=args.dilate_iters,
            iou_threshold=args.iou_threshold,
            use_points=args.use_points,
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
                        "medsam_checkpoint": None if args.skip_medsam else args.medsam_checkpoint,
                        "skip_medsam": args.skip_medsam,
                        "prompt": args.prompt, "use_points": args.use_points,
                        "dilate_iters": args.dilate_iters,
                        "iou_threshold": args.iou_threshold,
                    },
                    "cases": case_results,
                },
                fp, indent=2,
            )
        print(f"\nSaved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
