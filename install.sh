#!/usr/bin/env bash
# install.sh — Setup MSVM-UNet environment from scratch
# Tested: Ubuntu 22, CUDA 11.8–12.x, Python 3.8
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/venv}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
SYNAPSE_GDRIVE_ID="1BvpY0g9mKkkhdHpAX1HqDw8iTJNbFuwq"
CKPT_URL="https://github.com/gndlwch2w/msvm-unet/releases/download/v0.0.1/epoch.259-val_mean_dice.0.8500.ckpt"
CKPT_DIR="$SCRIPT_DIR/log/msvm_unet-synapse-r0/checkpoints"
CKPT_NAME="epoch.259-val_mean_dice.0.8500.ckpt"
DATA_DIR="${DATA_DIR:-/data}"
SYNAPSE_GDRIVE_ID="1BvpY0g9mKkkhdHpAX1HqDw8iTJNbFuwq"
SAMMED2D_GDRIVE_ID="1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl"
SAMMED2D_CKPT_DIR="$SCRIPT_DIR/SAM-Med2D/pretrain_model"
SAMMED2D_CKPT_NAME="sam-med2d_b.pth"

# ── helpers ──────────────────────────────────────────────────────────────────

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

# ── step 1: Python 3.8 ───────────────────────────────────────────────────────

info "Step 1: Checking Python 3.8"
if ! command -v python3.8 &>/dev/null; then
    info "Installing Python 3.8..."
    apt-get update -qq
    apt-get install -y python3.8 python3.8-venv python3.8-dev
else
    info "Python 3.8 already installed: $(python3.8 --version)"
fi

# ── step 2: GCC 11 ───────────────────────────────────────────────────────────

info "Step 2: Checking GCC 11 (required for selective_scan CUDA kernels)"
if ! command -v gcc-11 &>/dev/null; then
    info "Installing GCC 11..."
    apt-get update -qq
    apt-get install -y gcc-11 g++-11
else
    info "GCC 11 already installed: $(gcc-11 --version | head -1)"
fi

# ── step 3: Virtual environment ───────────────────────────────────────────────

info "Step 3: Creating venv at $VENV_DIR"
python3.8 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip wheel setuptools packaging --cache-dir "$PIP_CACHE_DIR" -q

# ── step 4: Detect CUDA and pick PyTorch wheel ───────────────────────────────

info "Step 4: Detecting CUDA version"
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
    info "Detected CUDA $CUDA_VER"

    if   [[ "$CUDA_MAJOR" -eq 11 && "$CUDA_MINOR" -eq 8 ]]; then
        TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_SUFFIX="cu118"
    elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -le 1 ]]; then
        TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_SUFFIX="cu121"
    elif [[ "$CUDA_MAJOR" -eq 12 ]]; then
        # CUDA 12.2–12.x → use cu121 wheels (forward-compatible)
        TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_SUFFIX="cu121"
    else
        warn "Unknown CUDA version $CUDA_VER, falling back to CPU-only PyTorch"
        TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cpu"
        TORCH_SUFFIX="cpu"
    fi
else
    warn "nvcc not found — installing CPU-only PyTorch"
    TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_SUFFIX="cpu"
fi

# ── step 5: Install PyTorch ───────────────────────────────────────────────────

info "Step 5: Installing PyTorch 2.1.0 ($TORCH_SUFFIX)"
pip install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url "$TORCH_EXTRA_INDEX" \
    --cache-dir "$PIP_CACHE_DIR"

# ── step 6: selective_scan CUDA kernels ───────────────────────────────────────

info "Step 6: Installing selective_scan CUDA kernel (GCC 11)"
cd "$SCRIPT_DIR/kernels/selective_scan"
CC=gcc-11 CXX=g++-11 pip install -e . --cache-dir "$PIP_CACHE_DIR"
cd "$SCRIPT_DIR"

# ── step 7: requirements.txt ─────────────────────────────────────────────────

info "Step 7: Installing requirements.txt"
# lightning==1.9.2 in requirements.txt pulls torch 1.13 — skip it here,
# install lightning 2.x explicitly after so the project's lightning.pytorch
# API calls work correctly.
grep -v "^lightning==" "$SCRIPT_DIR/requirements.txt" > /tmp/requirements_filtered.txt
pip install -r /tmp/requirements_filtered.txt --cache-dir "$PIP_CACHE_DIR"

# Reinstall PyTorch in case requirements.txt downgraded it
info "Step 7b: Re-pinning PyTorch 2.1.0 after requirements install"
pip install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url "$TORCH_EXTRA_INDEX" \
    --cache-dir "$PIP_CACHE_DIR"

# ── step 8: Lightning 2.x ────────────────────────────────────────────────────

info "Step 8: Installing Lightning >= 2.0 (project uses lightning.pytorch API)"
pip install "lightning>=2.0.0,<3.0.0" --cache-dir "$PIP_CACHE_DIR"

# ── step 9: medpy ────────────────────────────────────────────────────────────

info "Step 9: Installing medpy 0.5.2 (--no-deps)"
pip install medpy==0.5.2 --no-deps --cache-dir "$PIP_CACHE_DIR"

# ── step 10: reinstall selective_scan (editable link may break after upgrades) ─

info "Step 10: Verifying selective_scan editable install"
cd "$SCRIPT_DIR/kernels/selective_scan"
CC=gcc-11 CXX=g++-11 pip install -e . --cache-dir "$PIP_CACHE_DIR" -q
cd "$SCRIPT_DIR"

# ── step 11: Download Synapse dataset ────────────────────────────────────────

info "Step 11: Synapse dataset → $DATA_DIR/Synapse"
if [[ -d "$DATA_DIR/Synapse/train_npz" && -d "$DATA_DIR/Synapse/test_vol_h5" ]]; then
    info "Synapse dataset already exists, skipping download."
else
    mkdir -p "$DATA_DIR"

    # gdown cần có trong venv (đã active từ bước 3)
    pip install gdown -q --cache-dir "$PIP_CACHE_DIR"

    TMP_ZIP="$DATA_DIR/synapse_tmp.zip"
    TMP_EXTRACT="$DATA_DIR/synapse_tmp_extract"

    info "Downloading Synapse dataset (~938MB)..."
    gdown "$SYNAPSE_GDRIVE_ID" -O "$TMP_ZIP"

    info "Extracting Synapse folder..."
    apt-get install -y unzip -qq
    unzip "$TMP_ZIP" "project_TransUNet/data/Synapse/*" -d "$TMP_EXTRACT"

    mv "$TMP_EXTRACT/project_TransUNet/data/Synapse" "$DATA_DIR/Synapse"

    rm -rf "$TMP_ZIP" "$TMP_EXTRACT"
    info "Synapse dataset ready: $(ls "$DATA_DIR/Synapse")"
fi

# ── step 12: Download model checkpoint ───────────────────────────────────────

info "Step 12: Model checkpoint → $CKPT_DIR/$CKPT_NAME"
if [[ -f "$CKPT_DIR/$CKPT_NAME" ]]; then
    info "Checkpoint already exists, skipping download."
else
    mkdir -p "$CKPT_DIR"
    wget -q --show-progress -O "$CKPT_DIR/$CKPT_NAME" "$CKPT_URL"
    info "Checkpoint downloaded: $(du -sh "$CKPT_DIR/$CKPT_NAME" | cut -f1)"
fi

# ── step 13: SAM-Med2D checkpoint ────────────────────────────────────────────

info "Step 13: SAM-Med2D checkpoint → $SAMMED2D_CKPT_DIR/$SAMMED2D_CKPT_NAME"

# albumentations is required by SAM-Med2D's predictor (predictor_sammed.py)
pip install albumentations -q --cache-dir "$PIP_CACHE_DIR"

if [[ -f "$SAMMED2D_CKPT_DIR/$SAMMED2D_CKPT_NAME" ]]; then
    info "SAM-Med2D checkpoint already exists, skipping download."
else
    mkdir -p "$SAMMED2D_CKPT_DIR"

    # gdown cần có trong venv (đã active từ bước 3)
    pip install gdown -q --cache-dir "$PIP_CACHE_DIR"

    info "Downloading SAM-Med2D checkpoint (~2.4GB)..."
    gdown "$SAMMED2D_GDRIVE_ID" -O "$SAMMED2D_CKPT_DIR/$SAMMED2D_CKPT_NAME"
    info "SAM-Med2D checkpoint downloaded: $(du -sh "$SAMMED2D_CKPT_DIR/$SAMMED2D_CKPT_NAME" | cut -f1)"
fi

# ── done: smoke test ─────────────────────────────────────────────────────────

info "Smoke test..."
python - <<'PYEOF'
import torch, torchvision, timm, monai
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import einops, h5py, cv2, scipy, medpy
import albumentations
import selective_scan_cuda_oflex
import selective_scan_cuda_core

print(f"  torch:       {torch.__version__} | CUDA: {torch.cuda.is_available()}")
print(f"  torchvision: {torchvision.__version__}")
print(f"  timm:        {timm.__version__}")
print(f"  monai:       {monai.__version__}")
print(f"  lightning:   {L.__version__}")
print(f"  einops / h5py / opencv / scipy / medpy / albumentations: OK")
print(f"  selective_scan_cuda_oflex / core: OK")
PYEOF

echo ""
info "Installation complete!"
info "Activate the environment with: source $VENV_DIR/bin/activate"
info "Synapse dataset: $DATA_DIR/Synapse"
info "SAM-Med2D checkpoint: $SAMMED2D_CKPT_DIR/$SAMMED2D_CKPT_NAME"
