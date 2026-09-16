"""
corruptions.py — Synthetic CT-slice corruptions for zero-shot robustness eval.

All corruptions operate on a single 2D slice, float32 in [0, 1] (matching the
Synapse .npy.h5 storage format — verified, not assumed). Must be applied at
native resolution, before the model's own 512->224 resize/normalize — see
ROBUSTNESS_EVAL_PLAN.md, "Lưu ý khi implement".

Severity is an int level 0..MAX_SEVERITY; severity 0 is always a no-op
(clean-baseline sanity check). contrast_shift is a deterministic function of
severity only (no RNG), so every slice of a volume gets the identical shift
at a given severity level — satisfying "1 shift dùng chung cho cả volume"
without needing any volume-level state.
"""
import cv2
import numpy as np

MAX_SEVERITY = 4

# severity -> parameter; index 0 is always the clean/no-op level.
# Placeholder values — to be confirmed/tuned via the tier-0 visual/histogram
# smoke test (ROBUSTNESS_EVAL_PLAN.md checklist).
_GAUSSIAN_NOISE_STD   = [0.0, 0.02, 0.05, 0.08, 0.12]
_POISSON_SCALE        = [None, 200, 80, 40, 20]     # lower = fewer photons = more noise
_GAUSSIAN_BLUR_KERNEL = [0, 5, 9, 15, 21]           # 0 = no-op
_CONTRAST_FACTOR      = [1.0, 0.85, 0.7, 0.55, 0.4]  # <1 = reduced contrast


def gaussian_noise(slc: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Electronic sensor noise. Independent draw per slice."""
    std = _GAUSSIAN_NOISE_STD[severity]
    if std == 0.0:
        return slc
    noisy = slc + rng.normal(0.0, std, size=slc.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def poisson_noise(slc: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Quantum/photon noise (low-dose CT proxy). Independent draw per slice."""
    scale = _POISSON_SCALE[severity]
    if scale is None:
        return slc
    noisy = rng.poisson(np.clip(slc, 0.0, 1.0) * scale).astype(np.float32) / scale
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def gaussian_blur(slc: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Reconstruction/defocus blur. Deterministic given severity — rng unused,
    kept for a uniform (slc, severity, rng) interface across all corruptions."""
    k = _GAUSSIAN_BLUR_KERNEL[severity]
    if k == 0:
        return slc
    return cv2.GaussianBlur(slc, (k, k), 0).astype(np.float32)


def contrast_shift(slc: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Scanner calibration drift. Deterministic given severity (no RNG) — the
    same severity level always yields the same shift, so applying it slice by
    slice already gives every slice of a volume an identical shift."""
    factor = _CONTRAST_FACTOR[severity]
    if factor == 1.0:
        return slc
    shifted = (slc.astype(np.float32) - 0.5) * factor + 0.5
    return np.clip(shifted, 0.0, 1.0).astype(np.float32)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "poisson_noise": poisson_noise,
    "gaussian_blur": gaussian_blur,
    "contrast_shift": contrast_shift,
}


def apply_corruption(slc: np.ndarray, name: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    assert name in CORRUPTIONS, f"Unknown corruption {name!r}, choices: {list(CORRUPTIONS)}"
    assert 0 <= severity <= MAX_SEVERITY, f"severity must be in [0, {MAX_SEVERITY}]"
    return CORRUPTIONS[name](slc, severity, rng)


def corrupt_volume(volume: np.ndarray, name: str, severity: int, seed: int = 42) -> np.ndarray:
    """Apply a named corruption slice-by-slice to a whole [D, H, W] volume, at
    its native resolution. Call this on the raw volume right after
    load_volume() — i.e. *before* predict_volume()'s internal resize/
    normalize — so severity means the same thing regardless of the model's
    224x224 patch size. severity=0 is a no-op: returns the input unchanged,
    so it exactly reproduces the clean baseline."""
    if severity == 0:
        return volume
    rng = np.random.default_rng(seed)
    out = np.empty_like(volume)
    for d in range(volume.shape[0]):
        out[d] = apply_corruption(volume[d], name, severity, rng)
    return out
