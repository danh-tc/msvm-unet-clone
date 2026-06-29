#!/usr/bin/env python3
import argparse
import os
import os.path as osp
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from data import DATASETS
except Exception:
    DATASETS = {}


COLORS = np.array(
    [
        [0, 0, 0],
        [30, 144, 255],
        [0, 255, 0],
        [255, 0, 0],
        [0, 255, 255],
        [255, 0, 255],
        [255, 255, 0],
        [128, 0, 255],
        [255, 128, 0],
        [165, 42, 42],
        [160, 82, 45],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 0],
    ],
    dtype=np.uint8,
)

CLASS_NAMES = {
    "synapse": {
        0: "Background",
        1: "Aorta",
        2: "Gallbladder",
        3: "Left kidney",
        4: "Right kidney",
        5: "Liver",
        6: "Pancreas",
        7: "Spleen",
        8: "Stomach",
    },
    "acdc": {
        0: "Background",
        1: "Right ventricle",
        2: "Myocardium",
        3: "Left ventricle",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize medical segmentation samples stored as .npz, .npy, or .h5/.npy.h5."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=str, help="Path to a .npz, .npy, .h5, or .npy.h5 sample.")
    src.add_argument("--dataset", type=str, help="Dataset key from data.DATASETS, e.g. synapse.")
    parser.add_argument("--base-dir", type=str, default=None, help="Dataset root directory.")
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "valid", "test"))
    parser.add_argument("--index", type=int, default=0, help="Index in the split list when using --dataset.")
    parser.add_argument("--name", type=str, default=None, help="Filename/case name in the split list when using --dataset.")
    parser.add_argument("--list-dir", type=str, default=None, help="Override list directory when using --dataset.")
    parser.add_argument("--image-key", type=str, default="image")
    parser.add_argument("--label-key", type=str, default="label")
    parser.add_argument(
        "--slice",
        type=str,
        default="auto",
        help="Slice for 3D volumes: integer, middle, largest-label, or auto. Default: auto.",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity for labels.")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path.")
    parser.add_argument(
        "--report",
        action="store_true",
        help="For 3D volumes, create a report figure with multiple 2D slices covering foreground classes.",
    )
    parser.add_argument(
        "--max-report-slices",
        type=int,
        default=8,
        help="Maximum slices in --report mode. Default: 8.",
    )
    parser.add_argument(
        "--target-classes",
        type=str,
        default="auto",
        help="Class ids to cover in --report mode, e.g. 1,2,3,4,5,6,7,8. Default: auto foreground classes.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="auto",
        choices=("auto", "synapse", "acdc", "none"),
        help="Class name mapping for printed statistics.",
    )
    return parser.parse_args()


def get_font(size: int = 16) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_sample(path: str, image_key: str, label_key: str) -> Tuple[np.ndarray, np.ndarray]:
    suffixes = "".join(Path(path).suffixes)
    if path.endswith(".h5") or path.endswith(".npy.h5"):
        import h5py

        with h5py.File(path, "r") as data:
            return data[image_key][:], data[label_key][:]
    if path.endswith(".npz"):
        data = np.load(path)
        return data[image_key], data[label_key]
    if path.endswith(".npy"):
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        if isinstance(data, dict):
            return data[image_key], data[label_key]
        raise ValueError(f"{path} is a plain .npy array, so it does not contain '{image_key}' and '{label_key}'.")
    raise ValueError(f"Unsupported sample format: {path} ({suffixes})")


def read_list_item(list_dir: str, split: str, index: int) -> str:
    list_path = osp.join(list_dir, f"{split}.txt")
    if not osp.exists(list_path) and split == "valid":
        list_path = osp.join(list_dir, "val.txt")
    with open(list_path, "r", encoding="utf-8") as fp:
        names = [line.strip() for line in fp if line.strip()]
    if index < 0 or index >= len(names):
        raise IndexError(f"index {index} out of range for {list_path}, length={len(names)}")
    return names[index]


def default_base_dir(dataset: str) -> Optional[str]:
    if dataset in DATASETS:
        env_root = osp.expandvars(osp.join("$DATASET_HOME", DATASETS[dataset]["root_suffix"]))
        if "$" not in env_root and osp.exists(env_root):
            return env_root
    fallback = osp.join("data", "Synapse" if dataset.lower() == "synapse" else dataset)
    if osp.exists(fallback):
        return fallback
    return None


def resolve_dataset_path(args: argparse.Namespace) -> str:
    dataset = args.dataset.lower()
    if dataset not in DATASETS and args.list_dir is None:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Provide --list-dir and --base-dir for custom data.")

    base_dir = args.base_dir or default_base_dir(dataset)
    if base_dir is None:
        raise ValueError("Could not infer dataset root. Pass --base-dir explicitly.")

    list_dir = args.list_dir or DATASETS[dataset]["list_dir"]
    name = args.name or read_list_item(list_dir, args.split, args.index)
    split = "val" if args.split == "valid" else args.split

    candidates = [
        osp.join(base_dir, split, name),
        osp.join(base_dir, split, name + ".npz"),
        osp.join(base_dir, split, name + ".npy.h5"),
        osp.join(base_dir, "train_npz", name),
        osp.join(base_dir, "train_npz", name + ".npz"),
        osp.join(base_dir, "test_vol_h5", name),
        osp.join(base_dir, "test_vol_h5", name + ".npy.h5"),
    ]
    for path in candidates:
        if osp.exists(path):
            return path
    raise FileNotFoundError("Could not find sample. Tried:\n" + "\n".join(candidates))


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3]
        lo, hi = np.percentile(image, (1, 99))
        image = np.clip((image - lo) / (hi - lo + 1e-8), 0, 1)
        return (image * 255).astype(np.uint8)

    lo, hi = np.percentile(image, (1, 99))
    image = np.clip((image - lo) / (hi - lo + 1e-8), 0, 1)
    image = (image * 255).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=-1)


def colorize_label(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label).astype(np.int64)
    palette = COLORS
    if label.max(initial=0) >= len(palette):
        extra = np.random.default_rng(0).integers(0, 255, size=(label.max() - len(palette) + 1, 3), dtype=np.uint8)
        palette = np.vstack([palette, extra])
    return palette[np.clip(label, 0, len(palette) - 1)]


def draw_contours(image_rgb: np.ndarray, label: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    for cls_id in np.unique(label):
        if cls_id == 0:
            continue
        mask = (label == cls_id).astype(np.uint8)
        padded = np.pad(mask, 1, mode="constant")
        center = padded[1:-1, 1:-1]
        boundary = (
            (center != padded[:-2, 1:-1])
            | (center != padded[2:, 1:-1])
            | (center != padded[1:-1, :-2])
            | (center != padded[1:-1, 2:])
        ) & (center > 0)
        out[boundary] = COLORS[int(cls_id) % len(COLORS)]
    return out


def pick_slice(image: np.ndarray, label: np.ndarray, mode: str) -> Optional[int]:
    if image.ndim != 3 or label.ndim != 3:
        return None
    if mode == "auto":
        mode = "largest-label"
    if mode == "middle":
        return image.shape[0] // 2
    if mode == "largest-label":
        areas = (label > 0).reshape(label.shape[0], -1).sum(axis=1)
        return int(np.argmax(areas)) if areas.max(initial=0) > 0 else image.shape[0] // 2
    return int(mode)


def make_panel(image: np.ndarray, label: np.ndarray, alpha: float) -> np.ndarray:
    image_rgb = normalize_image(image)
    label_rgb = colorize_label(label)
    overlay = image_rgb.copy()
    fg = label > 0
    overlay[fg] = ((1 - alpha) * overlay[fg] + alpha * label_rgb[fg]).astype(np.uint8)
    contour = draw_contours(image_rgb, label)

    divider = np.full((image_rgb.shape[0], 6, 3), 255, dtype=np.uint8)
    return np.hstack([image_rgb, divider, label_rgb, divider, overlay, divider, contour])


def make_panel_with_titles(
    image: np.ndarray,
    label: np.ndarray,
    alpha: float,
    title: str,
    subtitle: str,
) -> Image.Image:
    panel = Image.fromarray(make_panel(image, label, alpha))
    font = get_font(16)
    small_font = get_font(13)
    header_h = 58
    out = Image.new("RGB", (panel.width, panel.height + header_h), "white")
    draw = ImageDraw.Draw(out)
    draw.text((8, 6), title, fill=(0, 0, 0), font=font)
    draw.text((8, 30), subtitle, fill=(60, 60, 60), font=small_font)

    col_w = image.shape[1]
    x_positions = [0, col_w + 6, 2 * col_w + 12, 3 * col_w + 18]
    for x, name in zip(x_positions, ("CT slice", "GT mask", "GT overlay", "GT contour")):
        draw.text((x + 8, header_h - 20), name, fill=(0, 0, 0), font=small_font)
    out.paste(panel, (0, header_h))
    return out


def parse_target_classes(value: str, label: np.ndarray) -> List[int]:
    if value != "auto":
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    classes = sorted(int(v) for v in np.unique(label) if int(v) != 0)
    return classes


def classes_present_text(label_slice: np.ndarray, class_names: Dict[int, str]) -> str:
    present = [int(v) for v in np.unique(label_slice) if int(v) != 0]
    if not present:
        return "GT: no foreground"
    names = [f"{i} {class_names.get(i, f'class_{i}')}" for i in present]
    return "GT: " + ", ".join(names)


def select_report_slices(label: np.ndarray, target_classes: Sequence[int], max_slices: int) -> List[int]:
    if label.ndim != 3:
        return []

    target = {int(c) for c in target_classes if np.any(label == int(c))}
    if not target:
        return [label.shape[0] // 2]

    flat = label.reshape(label.shape[0], -1)
    areas = {cls_id: (flat == cls_id).sum(axis=1) for cls_id in target}
    remaining = set(target)
    selected: List[int] = []

    while remaining and len(selected) < max_slices:
        best_z = None
        best_score = None
        for z in range(label.shape[0]):
            if z in selected:
                continue
            covered = [cls_id for cls_id in remaining if areas[cls_id][z] > 0]
            if not covered:
                continue
            # Prefer slices that cover more remaining organs, then larger GT area.
            score = (len(covered), int(sum(areas[cls_id][z] for cls_id in covered)))
            if best_score is None or score > best_score:
                best_score = score
                best_z = z
        if best_z is None:
            break
        selected.append(int(best_z))
        remaining -= {cls_id for cls_id in remaining if areas[cls_id][best_z] > 0}

    if not selected:
        foreground = (label > 0).reshape(label.shape[0], -1).sum(axis=1)
        selected = [int(np.argmax(foreground)) if foreground.max(initial=0) > 0 else label.shape[0] // 2]
    return selected


def make_legend(class_names: Dict[int, str], class_ids: Sequence[int], width: int) -> Image.Image:
    font = get_font(14)
    title_font = get_font(17)
    row_h = 24
    ids = [int(i) for i in class_ids if int(i) != 0]
    height = 42 + max(1, len(ids)) * row_h
    legend = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(legend)
    draw.text((8, 8), "GT label legend", fill=(0, 0, 0), font=title_font)
    if not ids:
        draw.text((8, 36), "No foreground labels", fill=(70, 70, 70), font=font)
        return legend

    x = 8
    y = 38
    col_w = max(150, width // 4)
    for idx, cls_id in enumerate(ids):
        if idx > 0 and idx % 4 == 0:
            x = 8
            y += row_h
        color = tuple(int(v) for v in COLORS[cls_id % len(COLORS)])
        draw.rectangle((x, y + 3, x + 16, y + 19), fill=color, outline=(0, 0, 0))
        draw.text((x + 24, y + 2), f"{cls_id}: {class_names.get(cls_id, f'class_{cls_id}')}", fill=(0, 0, 0), font=font)
        x += col_w
    return legend


def make_volume_report(
    image: np.ndarray,
    label: np.ndarray,
    alpha: float,
    class_names: Dict[int, str],
    target_classes: Sequence[int],
    max_slices: int,
    sample_path: str,
) -> Tuple[Image.Image, List[int]]:
    if image.ndim != 3 or label.ndim != 3:
        panel = Image.fromarray(make_panel(image, label, alpha))
        return panel, []

    selected = select_report_slices(label, target_classes, max_slices)
    rows = []
    covered = set()
    for row_idx, z in enumerate(selected, start=1):
        view_label = label[z]
        present = [int(v) for v in np.unique(view_label) if int(v) != 0]
        covered.update(present)
        title = f"Slice {z} ({row_idx}/{len(selected)})"
        subtitle = classes_present_text(view_label, class_names)
        rows.append(make_panel_with_titles(image[z], view_label, alpha, title, subtitle))

    width = max(row.width for row in rows)
    target_present = [int(c) for c in target_classes if np.any(label == int(c))]
    legend = make_legend(class_names, sorted(target_present), width)

    title_font = get_font(18)
    small_font = get_font(14)
    header = Image.new("RGB", (width, 76), "white")
    draw = ImageDraw.Draw(header)
    draw.text((8, 8), "Synapse 3D volume with GT annotations", fill=(0, 0, 0), font=title_font)
    draw.text((8, 34), f"sample: {sample_path}", fill=(50, 50, 50), font=small_font)
    draw.text(
        (8, 54),
        f"selected slices: {', '.join(str(z) for z in selected)} | covered GT classes: {', '.join(str(c) for c in sorted(covered))}",
        fill=(50, 50, 50),
        font=small_font,
    )

    gap = 12
    height = header.height + legend.height + gap * (len(rows) + 1) + sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    canvas.paste(header, (0, y))
    y += header.height
    canvas.paste(legend, (0, y))
    y += legend.height + gap
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + gap
    return canvas, selected


def infer_class_names(args: argparse.Namespace, label: np.ndarray) -> Dict[int, str]:
    if args.class_names == "none":
        return {}
    if args.class_names in CLASS_NAMES:
        return CLASS_NAMES[args.class_names]
    if args.dataset and args.dataset.lower() in CLASS_NAMES:
        return CLASS_NAMES[args.dataset.lower()]

    max_label = int(np.max(label)) if label.size else 0
    if max_label <= 3:
        return CLASS_NAMES["acdc"]
    if max_label <= 8:
        return CLASS_NAMES["synapse"]
    return {}


def format_label_stats(label: np.ndarray, class_names: Dict[int, str], title: str) -> str:
    label = np.asarray(label).astype(np.int64)
    unique, counts = np.unique(label, return_counts=True)
    total = int(label.size)
    foreground = int(counts[unique != 0].sum()) if np.any(unique != 0) else 0
    organ_ids = [int(k) for k, v in zip(unique, counts) if int(k) != 0 and int(v) > 0]

    lines = [
        f"{title}",
        f"  total pixels/voxels: {total}",
        f"  foreground pixels/voxels: {foreground} ({foreground / total * 100:.4f}% of total)",
        f"  foreground classes present: {len(organ_ids)}"
        + (f" ({', '.join(str(i) for i in organ_ids)})" if organ_ids else ""),
    ]
    lines.append("  class breakdown:")
    lines.append("    id  name             count        %total     %foreground")
    for cls_id, count in zip(unique, counts):
        cls_id = int(cls_id)
        count = int(count)
        if cls_id == 0:
            continue
        pct_total = count / total * 100 if total else 0
        pct_fg = count / foreground * 100 if foreground else 0
        name = class_names.get(cls_id, f"class_{cls_id}")
        lines.append(f"    {cls_id:<3} {name:<16} {count:<12} {pct_total:>8.4f}% {pct_fg:>10.4f}%")
    if not organ_ids:
        lines.append("    no foreground label on this view")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sample_path = args.file or resolve_dataset_path(args)
    image, label = load_sample(sample_path, args.image_key, args.label_key)
    class_names = infer_class_names(args, label)

    if args.report:
        target_classes = parse_target_classes(args.target_classes, label)
        report, selected_slices = make_volume_report(
            image=image,
            label=label,
            alpha=args.alpha,
            class_names=class_names,
            target_classes=target_classes,
            max_slices=args.max_report_slices,
            sample_path=sample_path,
        )
        view_label = label[selected_slices[0]] if selected_slices else label
        z = selected_slices[0] if selected_slices else None
        panel_image = report
    else:
        z = pick_slice(image, label, args.slice)
        if z is not None:
            if z < 0 or z >= image.shape[0]:
                raise IndexError(f"slice {z} out of range for volume depth {image.shape[0]}")
            view_image = image[z]
            view_label = label[z]
        else:
            view_image = image
            view_label = label

        if view_image.shape[:2] != view_label.shape[:2]:
            raise ValueError(f"Image/label shape mismatch after slicing: {view_image.shape} vs {view_label.shape}")
        panel_image = Image.fromarray(make_panel(view_image, view_label, args.alpha))

    out = args.out
    if out is None:
        stem = Path(sample_path).name.replace(".npy.h5", "").replace(".npz", "").replace(".h5", "").replace(".npy", "")
        if args.report:
            suffix = "_report"
        else:
            suffix = f"_z{z:03d}" if z is not None else ""
        out = osp.join("visualizations", f"{stem}{suffix}.png")
    os.makedirs(osp.dirname(out) or ".", exist_ok=True)
    panel_image.save(out)

    unique, counts = np.unique(view_label, return_counts=True)
    label_stats = ", ".join(f"{int(k)}:{int(v)}" for k, v in zip(unique, counts))
    print(f"sample: {sample_path}")
    print(f"image shape: {image.shape}, dtype={image.dtype}, range=({float(np.min(image)):.4g}, {float(np.max(image)):.4g})")
    print(f"label shape: {label.shape}, dtype={label.dtype}, labels={label_stats}")
    if z is not None:
        print(f"visualized slice: {z}")
    print()
    stats_blocks = [format_label_stats(view_label, class_names, "Visible slice statistics")]
    if label.ndim == 3:
        stats_blocks.append(format_label_stats(label, class_names, "Whole volume statistics"))
    stats_text = "\n\n".join(stats_blocks)
    print(stats_text)
    stats_out = str(Path(out).with_suffix(".stats.txt"))
    with open(stats_out, "w", encoding="utf-8") as fp:
        fp.write(f"sample: {sample_path}\n")
        fp.write(f"image shape: {image.shape}, dtype={image.dtype}\n")
        fp.write(f"label shape: {label.shape}, dtype={label.dtype}\n")
        if z is not None:
            fp.write(f"visualized slice: {z}\n")
        fp.write("\n")
        fp.write(stats_text)
        fp.write("\n")
    print(f"saved: {out}")
    print(f"stats: {stats_out}")


if __name__ == "__main__":
    main()
