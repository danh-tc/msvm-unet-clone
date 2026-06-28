# A Local-Global Fusion Vision Mamba UNet Framework for Medical Image Segmentation

> **Journal:** Engineering Applications of Artificial Intelligence 169 (2026) 113987  
> **DOI:** https://doi.org/10.1016/j.engappai.2026.113987  
> **Code:** https://github.com/NicoleDyson/LGFVM-UNet  
> **Received:** 13 May 2025 | **Revised:** 3 January 2026 | **Accepted:** 27 January 2026

**Authors:** Yanbo Li, Zihan Mao, Feiwei Qin*, Yong Peng, Guodao Zhang, Xugang Xi (Hangzhou Dianzi University) · Xiaoqin Ma (Chizhou University) · Huanhuan Yu, Yu Zhou (Children's Hospital, Zhejiang University School of Medicine) · Zhu Zhu** (Third Affiliated Hospital of Wenzhou Medical University)

---

## Abstract

As a State Space Model (SSM) that achieves long-range dependency modeling with linear computational complexity, Mamba demonstrates significant efficiency advantages in medical image segmentation. However, while Mamba-based methods enable long-range modeling with linear complexity, their global dependency mechanisms often lead to local feature attenuation, particularly affecting the processing of complex anatomical structures. Existing multi-scale fusion methods also exhibit limited compatibility with State Space Models.

To address these challenges, this paper proposes the **Local-Global Fusion Vision Mamba UNet (LGFVM-UNet)** framework. Its core innovation lies in the **Dynamic Gating-enhanced Local-Global Fusion Visual State Space (LGF-VSS) block**, which enables the synergistic modeling of global context and local details. Additionally, we designed a **Multi-level Cross-scale Feature Fusion Block (MCFB)** that enhances multi-scale feature representation through bidirectional resampling and spatial-channel dual attention mechanisms. Furthermore, we propose a **Gradient Statistics-based Adaptive Hierarchical Loss** that dynamically adjusts multi-level supervision weights to optimize the learning process.

Results demonstrate that our approach outperforms state-of-the-art methods, excelling in long-range dependency modeling, local detail capture, and multi-scale feature fusion.

**Keywords:** Medical image segmentation · Vision state space model · Local-global feature fusion · Multi-scale feature integration · Adaptive loss weighting

---

## 1. Introduction

Medical image segmentation directly influences diagnostic accuracy and treatment planning. CNNs rely on local convolutional kernels limiting global modeling ability; Vision Transformers model long-range dependencies but with quadratic self-attention complexity.

State Space Models (SSMs), notably Mamba, offer global receptive fields with linear computational complexity. Despite this efficiency, directly applying Mamba to medical imaging reveals two critical limitations that motivate LGFVM-UNet:

1. **"Global Dominance" problem**: Mamba's process of flattening 2D images into 1D sequences dilutes spatial relationships between immediate neighbors. In clinical scenarios, local texture and edge information are paramount.

2. **Semantic gap between Mamba's linear scanning and decoder's spatial reconstruction**: Standard skip connections fail to align these representations, resulting in suboptimal information flow.

**Key innovations to address these conflicts:**

- **LGF-VSS Block**: Parallel multi-scale convolutional branches for enhanced local feature extraction + dynamic QuadGate mechanism for adaptive feature fusion
- **MCFB**: Replaces simple skip connections; leverages bidirectional resampling and spatial-channel dual attention to ensure encoder details are semantically aligned with decoder's global context
- **Gradient Statistics-based Adaptive Hierarchical Loss**: Monitors real-time gradient magnitudes to automatically adjust supervision weights and ensure the model focuses on the most difficult semantic levels

---

## 2. Related Work

### 2.1 Evolution of UNet Architecture

- **UNet++**: Nested and dense skip connections to reduce semantic gap between encoder and decoder sub-networks
- **Attention U-Net**: Gating signals in skip connections to suppress irrelevant regions
- **DDU-Net, DC-UNet, N-Net**: Address high-resolution inputs, dual-channel feature extraction, and dual-encoder designs

### 2.2 Application of Transformers in Medical Image Segmentation

- **TransUNet**: Embedded Transformers into the encoder to model global relationships via serialized image patches
- **Swin-UNet**: Window-based self-attention for improved efficiency
- **SSFormer**: Progressive Local Decoder with pyramid Transformer encoder
- **TransFuse**: Parallel dual-branch (CNN + Transformer)

### 2.3 Advances in State Space Models (SSMs)

- **VM-UNet**: Pioneered integration of VSS blocks with UNet architecture
- **VM-UNetV2**: Scan-Decoupled Integration (SDI) module for information exchange between scales
- **MSVM-UNet**: Multi-Scale Visual State Space blocks + Large Kernel Patch Embedding for directional sensitivity
- **H-vmunet**: Higher-order 2D Selective Scanning (H-SS2D) to reduce redundant scan paths
- **LightM-UNet**: Lightweight Mamba framework with Residual Vision Mamba Layers
- **LKM-UNet**: Large Mamba kernels with hierarchical bidirectional designs
- **CM-UNet**: Hybrid CNN-Mamba architecture where CNN encoder extracts local features and Mamba decoder aggregates global information

### 2.4 Learning-based and Anatomy-aware Segmentation Strategies

- **CASTformer**: Class-aware adversarial framework with learnable class embeddings as query vectors
- **ACTION / ACTION++**: Anatomy-aware contrastive distillation with adaptive temperature coefficients for class imbalance
- **ARCO**: Gradient variance reduction via hierarchical group sampling strategy
- **MORSE**: Implicit neural representations with stochastic experts for continuous anatomical rendering at sub-pixel level

---

## 3. Method

### 3.1 Overall Architecture of LGFVM-UNet

LGFVM-UNet follows a classic U-shaped topology:

```
Input X ∈ ℝ^{H×W×3}
    │
    ├── Patch Embedding
    │
    ├── Encoder (4 stages: H/4, H/8, H/16, H/32)
    │   ├── Stage 1: LGF-VSS Block × 2
    │   ├── Stage 2: Patch Merging → LGF-VSS Block × 2
    │   ├── Stage 3: Patch Merging → LGF-VSS Block × 9  ← bottleneck
    │   └── Stage 4: Patch Merging → LGF-VSS Block × 2
    │
    ├── Decoder (4 stages, mirrored)
    │   ├── MCFB (Multi-level Cross-scale Feature Fusion Block)
    │   ├── LGF-VSS Block processing at each level
    │   └── Patch Expanding for upsampling
    │
    └── Final 1×1 Conv → Segmentation mask
```

**Key distinction from MSVM-UNet**: encoder and decoder are connected not by simple skip connections but by **MCFB**, which enables bidirectional information flow and cross-scale feature alignment.

**Default configuration:** encoder depth `[2,2,2,2]` + decoder depth `[2,2,2,1]`, kernel sizes `[1,3,5]`, adaptive loss hyperparameters α=0.8, τ=0.2, λ=0.75

---

### 3.2 Local-Global Fusion Visual State Space (LGF-VSS) Block

The LGF-VSS block captures two complementary types of information simultaneously:

- **Type 1 — High-frequency details** (texture, edges, small anatomical anomalies): via convolutional operations
- **Type 2 — Low-frequency context** (organ placement, relative geometry, long-range dependencies): via Mamba branch

#### 3.2.1 QuadGate Dynamic Fusion Mechanism

For a processed feature map $\mathcal{F}_i^1$ at stage $i$, first apply normalization and activation:

$$\mathcal{F}_i^1 = \sigma\left(Conv_{3\times3}(LN(\mathcal{F}_i^0))\right) \tag{1}$$

The **QuadGate module** generates pixel-wise data-dependent weights:

$$\mathcal{G} = \text{Softmax}\left(Conv_{1\times1}\left(\sigma\left(\text{Linear}(\text{AvgPool}(\mathcal{F}_i^1))\right)\right)\right) \tag{2}$$

- AvgPool reduces spatial resolution to 1×1 while preserving channel dimensions
- Linear projection facilitates cross-channel interaction without altering tensor shape
- SiLU activation for nonlinear expressivity
- $Conv_{1\times1}$ expands channel dimensions by 4× before channel-wise Softmax

$\mathcal{G}$ is split into four components $\{w_1, w_3, w_5, w_m\}$ corresponding to weights for 1×1, 3×3, 5×5 convolutional branches and the Mamba state space branch.

#### 3.2.2 LGF-SS2D Fusion

The fused feature map:

$$\mathcal{F}_{fused} = w_m \odot SS2D(\mathcal{F}_i^1) + \sum_{k \in \{1,3,5\}} w_k \odot BN\left(Conv_{k\times k}(\mathcal{F}_i^1)\right) \tag{3}$$

where $\odot$ denotes broadcasted element-wise multiplication.

**SS2D operation**: generates four distinct flattened sequences by traversing spatial dimensions from four corners (top-left, top-right, bottom-left, bottom-right). Each spatial location is processed in a unique sequential order to capture comprehensive spatial dependencies.

#### 3.2.3 Dual Residual Connections

$$\mathcal{F}_i^2 = LN\left(\mathcal{F}_i^1 + \mathcal{F}_{fused}\right) \tag{4}$$

$$\mathcal{F}_i^3 = \mathcal{F}_i^2 \odot \sigma\left(LN(\mathcal{F}_i^0)\right) + \mathcal{F}_i^0 \tag{5}$$

- **First residual** (Eq. 4): combines preprocessed features with fused features after layer normalization
- **Second residual** (Eq. 5): gated residual path that modulates features based on the normalized input — facilitates multi-level feature integration and optimizes gradient propagation

#### Comparison with MSVM-UNet's MSVSS Block

| Aspect | MSVM-UNet (MSVSS) | LGFVM-UNet (LGF-VSS) |
|--------|-------------------|-----------------------|
| Fusion strategy | Sequential (Mamba → FFN with conv) | Parallel branches with dynamic gating |
| Weight assignment | Fixed equal weights per kernel | Pixel-wise learned weights (QuadGate) |
| Residual design | Single standard residual | Dual residual with gated path |
| Local-global arbitration | Implicit (architectural ordering) | Explicit (data-dependent gate) |

---

### 3.3 Multi-level Cross-scale Feature Fusion Block (MCFB)

The MCFB replaces horizontal skip connections. Its primary role is **semantic alignment** between encoder and decoder features.

**Problem it solves**: In Mamba-based networks, encoder features $F_{enc}$ undergo selective scanning, transforming spatial topology differently than convolutional pooling. Simple concatenation assumes spatial coherence that may not exist.

#### 3.3.1 Cross-Scale Attention (CSA) Module

For the *i*-th decoder stage, MCFB receives feature maps from **all** encoder stages $\{F_j | j=1,2,3,4\}$ and current decoder output $D_i$.

**Step 1 — Dynamic scale alignment to current decoder resolution:**

$$F_j' = \begin{cases} D_{i-j}(F_j), & j < i \quad \text{(downsampling with stride } 2^{i-j}) \\ U_{k-i}(F_k), & k > i \quad \text{(bilinear upsampling with scale } 2^{k-1}) \end{cases} \tag{6}$$

**Step 2 — Spatial attention (correlates decoder and encoder features):**

$$Q = BN(Conv_{1\times1}(D_i)), \quad K = BN(Conv_{3\times3}(F_j')) \tag{7}$$
$$\Psi_{spatial} = \text{Sigmoid}(Q \oplus K) \in [0,1]^{H_i \times W_i} \tag{8}$$

**Step 3 — Channel attention (squeeze-excitation):**

$$\Psi_{channel} = \text{Sigmoid}\left(Conv_{1\times1}\left(\delta\left(Conv_{1\times1}(\text{AvgPool}(\Psi_{spatial}))\right)\right)\right) \in [0,1]^{C_i/2} \tag{9}$$

**Step 4 — Combined attention and refinement:**

$$\mathcal{T}_{att} = \text{Sigmoid}\left(BN\left(Conv_{1\times1}(\Psi_{spatial} \odot \Psi_{channel})\right)\right)$$
$$C_i = \mathcal{T}_{att} \odot F_j' \tag{10}$$

For higher-level features ($k > i$), apply additional upsampling: $C_k' = U_{k-i}(Conv_{3\times3}(C_k))$.

**Step 5 — Final fusion with decoder features:**

$$\mathcal{F}_{concat} = \text{Concat}\left([C_j, C_i, C_k']_{\forall j<i, k>i}\right)$$
$$\mathcal{F}_{fused} = \text{LGFVSS}(\mathcal{F}_{concat})$$
$$D_{i+1} = D_i + \mathcal{F}_{fused} \tag{11}$$

This allows the decoder to access the **entire encoding history**, not just features from a single corresponding level — effectively mitigating the semantic gap between 1D Mamba scanning and 2D reconstruction.

#### Ablation: MCFB vs Standard Skip Connections

| Configuration | DSC (%) | HD95 (mm) | FLOPs (G) | Params (M) |
|---------------|---------|-----------|-----------|------------|
| No LGF-VSS, No MCFB | 84.22 | 8.22 | 12.63 | 22.04 |
| LGF-VSS only | 85.52 | 7.48 | 2.56 | 19.12 |
| MCFB only | 87.65 | 7.02 | 26.78 | 45.33 |
| **LGF-VSS + MCFB (full)** | **88.74** | **6.65** | 14.03 | 41.68 |

---

### 3.4 Gradient Statistics-based Adaptive Hierarchical Loss

**Problem with standard deep supervision**: static, pre-defined weights cannot adapt to the shifting focus of learning across different scales. Early training emphasizes coarse, high-level features; later epochs focus on fine details.

**Mechanism**: Monitor real-time gradient magnitudes at each supervised stage. If a stage becomes stuck or exhibits unstable oscillations (high gradient variance), its loss weight is automatically adjusted.

#### Gradient Magnitude Computation

For the *l*-th decoder stage at training step $t$:

$$g_l^{(t)} = \frac{1}{|\theta|} \sum_{i=1}^{|\theta|} \left| \frac{\partial \mathcal{L}_l^{(t)}}{\partial \theta_i} \right| \tag{12}$$

#### Momentum-based Weight Smoothing

$$w_l^{(t)} = \alpha \cdot w_l^{(t-1)} + (1-\alpha) \cdot \frac{\exp(g_l^{(t)}/\tau)}{\sum_{k=1}^L \exp(g_k^{(t)}/\tau)} \tag{13}$$

- $\alpha \in [0,1]$: momentum coefficient controlling historical information retention (optimal: **α = 0.8**)
- $\tau > 0$: temperature parameter controlling weight distribution sharpness (optimal: **τ = 0.2**)

#### Final Composite Loss

$$\mathcal{L}_{total}^{(t)} = \underbrace{\sum_{l=1}^{L} w_l^{(t)} \cdot \mathcal{L}_l^{(t)}}_{\text{adaptive supervision}} + \lambda \cdot \underbrace{\mathcal{L}_{final}^{(t)}}_{\text{main output}} \tag{14}$$

- Supervision applied at first three decoder stages only; final stage contributes directly to main output loss
- Optimal: **λ = 0.75**

**Implementation techniques:**
1. Gradient Normalization: Z-score standardization to mitigate magnitude variations across layers
2. Gradient Clipping: threshold $g_{clip} = 1.0$ prevents exploding gradients
3. Hybrid Loss Function: combination of Binary Cross-Entropy (BCE) and Dice loss
4. Real-time Adjustment: gradient hooks during backward pass enable immediate weight updates

---

## 4. Experiments

### 4.1 Datasets

| Dataset | Modality | Cases | Classes | Split |
|---------|----------|-------|---------|-------|
| **Synapse Multi-organ** | CT | 30 cases, 3779 slices | 8 organs | 18 train / 12 test |
| **ACDC Cardiac MRI** | MRI | 100 cases | 3 (RV, Myo, LV) | 70/10/20 |
| **ISIC2017 Dermoscopy** | Dermatoscopy | 2000 train / 600 test | skin lesion | 60/20/20 |
| **ISIC2018 Dermoscopy** | Dermatoscopy | 2594 images | 7 skin lesion types | 60/20/20 |
| **CVC-ClinicDB Colonoscopy** | Endoscopy | 612 polyp images | polyp | same partitioning |

### 4.2 Implementation Details

- **Input size**: 224×224 (Synapse, ACDC), 256×256 (ACDC official), 384×288 (ISIC)
- **Optimizer**: AdamW, weight decay 1e-4, initial lr 1e-4 with cosine annealing
- **Batch size**: 32, **max epochs**: 200, early stopping patience = 15
- **GPU**: NVIDIA GeForce RTX 4090

**Data Augmentation:**
1. Spatial Transformations: rotations at 90° multiples with axial flipping, or free rotations within ±15°; elastic deformations (30% probability)
2. Intensity Perturbations: window width/level shifts ±15%, local brightness modulation ±20%, constrained Gaussian noise (σ < 0.05)
3. Anatomical Integrity Preservation: third-order spline interpolation to maintain anatomical boundaries during transformations

---

## 5. Results

### 5.1 Synapse Multi-organ Segmentation (Table 1)

| Methods | DSC (%) ↑ | HD95 (mm) ↓ | Aorta | Gallbladder | Kidney(L) | Kidney(R) | Liver | Pancreas | Spleen | Stomach |
|---------|-----------|------------|-------|-------------|-----------|-----------|-------|----------|--------|---------|
| UNet | 71.98 | 22.39 | 83.72 | 37.90 | 89.07 | 80.74 | 87.75 | 39.77 | 89.57 | 67.34 |
| UNet++ | 77.70 | 27.58 | 85.95 | 40.18 | 90.51 | 88.33 | 94.46 | 48.53 | 91.54 | 82.11 |
| SwinUNet | 78.68 | 35.57 | 77.15 | 70.52 | 88.32 | 86.13 | 91.91 | 51.57 | 87.70 | 76.15 |
| TransUNet | 82.42 | 25.54 | 87.72 | **76.52** | 88.87 | 81.38 | 94.99 | 54.49 | 89.18 | 86.25 |
| Semi-Mamba-UNet | 83.49 | 26.91 | 87.21 | 70.14 | 89.82 | 86.15 | 95.18 | 62.08 | 92.18 | 85.16 |
| VM-UNetV2 | 83.56 | 22.29 | 83.01 | 71.68 | 90.17 | 89.30 | 93.49 | 64.81 | 91.82 | 84.57 |
| H-vmunet | 86.50 | 7.94 | 88.63 | 70.93 | 92.34 | 91.24 | 95.91 | 69.19 | 93.81 | 90.02 |
| MSVM-UNet | _87.41_ | 8.91 | _89.04_ | 70.45 | _93.77_ | _92.18_ | _96.39_ | 73.25 | _94.52_ | 89.68 |
| **LGFVM-UNet (ours)** | **88.74** | **6.65** | **89.13** | _75.18_ | **94.33** | **93.87** | **96.24** | **75.00** | **94.55** | **91.61** |

> Bold = best, Underline = second-best

**Key improvements over MSVM-UNet:**
- DSC: +1.33% (88.74 vs 87.41)
- HD95: -2.26mm (6.65 vs 8.91)
- Gallbladder: +4.73% (75.18 vs 70.45) — largest absolute gain
- Pancreas: +1.75% (75.00 vs 73.25)
- Stomach: +1.93% (91.61 vs 89.68)

### 5.2 ACDC Cardiac MRI (Table 2)

| Methods | DSC (%) ↑ | HD95 (mm) ↓ | RV | Myo | LV |
|---------|-----------|------------|-----|-----|-----|
| UNet | 80.25 | 49.73 | 74.46 | 76.39 | 89.88 |
| TransUNet | 88.69 | 35.90 | 86.40 | 85.88 | 93.80 |
| VM-UNetV2 | 91.03 | **19.87** | 90.46 | 87.67 | 94.95 |
| H-vmunet | 92.07 | 21.58 | 91.77 | 90.09 | 95.34 |
| MSVM-UNet | 91.76 | 30.60 | 91.13 | 88.80 | 95.35 |
| **LGFVM-UNet (ours)** | **92.40** | 21.97 | **92.19** | **90.13** | **95.71** |

### 5.3 Skin Lesion & Polyp Datasets (Tables 3–5)

**ISIC2017:** 91.70% DSC, 84.67% mIoU (best)  
**ISIC2018:** 91.35% DSC, 84.09% mIoU (best)  
**CVC-ClinicDB:** 91.03% DSC, 83.55% mIoU (best), 99.55% Specificity

### 5.4 Model Complexity (Table 6)

| Methods | DSC (%) | HD95 (mm) | Params (M) | FLOPs (G) |
|---------|---------|-----------|-----------|-----------|
| UNet | 71.98 | 22.39 | 31.04 | 41.91 |
| TransUNet | 82.42 | 25.54 | 111.42 | 43.12 |
| VM-UNetV2 | 83.56 | 22.29 | 22.77 | 2.37 |
| H-vmunet | 86.50 | 8.97 | 8.97 | 8.12 |
| MSVM-UNet | 87.41 | 8.91 | 35.93 | 13.90 |
| **LGFVM-UNet** | **88.74** | **6.65** | 41.68 | 14.03 |

> LGFVM-UNet: +1.33% DSC, -2.26mm HD95 vs MSVM-UNet, at cost of +5.75M params and +0.13G FLOPs

---

## 6. Ablation Studies

### 6.1 Impact of Key Components (Table 7)

| LGF-VSS | MCFB | DSC (%) | HD95 (mm) | FLOPs (G) | Params (M) |
|---------|------|---------|-----------|-----------|------------|
| ✗ | ✗ | 84.22 | 8.22 | 12.63 | 22.04 |
| ✓ | ✗ | 85.52 | 7.48 | 2.56 | 19.12 |
| ✗ | ✓ | 87.65 | 7.02 | 26.78 | 45.33 |
| **✓** | **✓** | **88.74** | **6.65** | 14.03 | 41.68 |

Both components contribute substantially. MCFB alone achieves 87.65% but at high computational cost (26.78G FLOPs). LGF-VSS + MCFB achieves the best tradeoff at 14.03G FLOPs.

### 6.2 Optimal Encoder/Decoder Block Distribution (Table 8)

| Encoder depth | Decoder depth | DSC (%) | HD95 (mm) | FLOPs (G) | Params (M) |
|---------------|---------------|---------|-----------|-----------|------------|
| [1,1,1,1] | [2,2,2,1] | 87.52 | 10.98 | 13.87 | 37.68 |
| [2,2,2,2] | [1,1,1,1] | 87.52 | 9.26 | 13.95 | 37.73 |
| [1,2,2,2] | [2,2,2,1] | 87.08 | 9.27 | 13.95 | 41.63 |
| **[2,2,2,2]** | **[2,2,2,1]** | **88.74** | **6.65** | 14.03 | 41.68 |
| [2,2,2,2] | [2,2,2,2] | 87.26 | 9.09 | 14.11 | 41.74 |
| [2,2,4,4] | [2,2,2,1] | 87.19 | 8.66 | 14.12 | 49.2 |
| [4,3,2,1] | [2,2,2,1] | 87.32 | **7.82** | 14.21 | 38.99 |
| [4,4,4,4] | [2,2,2,1] | 86.85 | 8.24 | 14.36 | 49.7 |

Key findings:
- Balanced [2,2,2,2] encoder achieves optimal DSC
- [4,3,2,1] progressively decreasing depth achieves best HD95 (7.82mm) — different optimal patterns for DSC vs boundary metrics
- Excessive depth (4,4,4,4) leads to feature over-smoothing and degrades DSC

### 6.3 LGF-VSS Internal Structure (Table 9)

| Configuration | SSM branch | Conv branch | DSC (%) | HD95 (mm) |
|---------------|-----------|-------------|---------|-----------|
| SS2D only (Standard VSS) | ✓ | ✗ | 84.56 | 8.15 |
| CNN only (ResNet-like) | ✗ | Multi-scale | 85.12 | 7.92 |
| Simplified hybrid | ✓ | Single 3×3 | 86.45 | 7.34 |
| **LGF-VSS (Ours)** | **✓** | **Multi-scale** | **88.74** | **6.65** |

SS2D alone (standard VSS) achieves only 84.56% — confirms pure Mamba struggles with local feature attenuation. Multi-scale CNN branches are essential.

### 6.4 Parallel Convolution Kernel Sizes (Table 10)

| Kernels | DSC (%) | HD95 (mm) | FLOPs (G) | Params (M) |
|---------|---------|-----------|-----------|------------|
| [1, 3] | 87.46 | 13.72 | 13.72 | 37.57 |
| [3, 5] | 88.56 | 7.81 | 13.97 | 37.74 |
| **[1, 3, 5]** | **88.74** | **6.65** | 14.03 | 41.68 |
| [3, 5, 7] | 88.49 | **6.25** | 14.52 | 42.03 |
| [1, 3, 5, 7] | 88.35 | 6.83 | 14.58 | 45.97 |

**[1,3,5]** achieves best DSC with reasonable parameters. Adding kernel 7 slightly improves HD95 but decreases DSC — diminishing returns beyond 3 kernels.

### 6.5 Fusion Strategy Comparison (Table 11)

| Fusion strategy | Modulation type | DSC (%) | HD95 (mm) |
|----------------|----------------|---------|-----------|
| Fixed summation | Static | 85.52 | 7.48 |
| Concatenate + 1×1 Conv | Learned (Fixed) | 86.88 | 7.12 |
| **QuadGate (Ours)** | **Dynamic (Pixel-wise)** | **88.74** | **6.65** |

QuadGate dynamic pixel-wise gating significantly outperforms fixed fusion strategies. By assigning higher weights to convolutional branches at organ boundaries and prioritizing Mamba in uniform tissue areas, QuadGate ensures optimal balance globally.

### 6.6 MCFB Component Analysis (Table 12)

| Configuration | Input level | Attention mechanism | DSC (%) | HD95 (mm) |
|---------------|------------|---------------------|---------|-----------|
| Baseline concatenation | Single-stage | None | 84.22 | 8.22 |
| Single-stage MCFB | Single-stage | Spatial-Channel | 86.15 | 7.45 |
| MCFB without attention | Multi-level | None | 87.34 | **6.98** |
| **MCFB (Ours)** | **Multi-level** | **Spatial-Channel** | **88.74** | **6.65** |

Multi-level access provides +1.19% DSC over single-stage. Attention mechanism provides additional +1.40% DSC.

### 6.7 Cross-scale Fusion Mechanism Comparison (Table 13)

| Fusion mechanism | DSC (%) | HD95 (mm) |
|-----------------|---------|-----------|
| Attention gate (Attention U-Net) | 85.34 | 7.85 |
| SDI module (VM-UNetV2) | 87.12 | 7.15 |
| **MCFB (Ours)** | **88.74** | **6.65** |

MCFB: +1.62% DSC over SDI, +3.40% DSC over standard attention gate.

### 6.8 Dynamic Gating vs Other Fusion Mechanisms (Table 14)

| Mechanism | Type | DSC (%) | HD95 (mm) |
|-----------|------|---------|-----------|
| Fixed summation | Static fusion | 85.52 | 7.48 |
| CBAM | Spatial-channel attention | 86.45 | 7.22 |
| GLSP module | Parallel perception | 87.80 | **6.95** |
| **QuadGate (Ours)** | **Dynamic Gating** | **88.74** | **6.65** |

### 6.9 Adaptive Loss Contribution (Table 15)

| Adaptive loss | DSC (%) | HD95 (mm) |
|---------------|---------|-----------|
| Disabled | 85.82 | 7.48 |
| **Enabled** | **88.74** | **6.65** |

Adaptive hierarchical loss contributes +2.92% DSC and -0.83mm HD95 — a substantial gain purely from training strategy.

### 6.10 Adaptive Loss Hyperparameters (Figure 6)

- **Momentum α**: Optimal at 0.8. Values below 0.6 cause excessive fluctuations; α=1.0 disables adaptation.
- **Temperature τ**: Optimal at 0.2. Higher τ makes weights more uniform, reducing effectiveness.
- **Main output weight λ**: Optimal at 0.75. Removing hierarchical supervision entirely degrades to 85.82%.

---

## 7. Qualitative Analysis & Failure Cases

**Successes:** LGFVM-UNet produces more accurate boundary delineation compared to both Transformer-based (TransUNet, SwinUNet) and other Mamba-based (VM-UNetV2, MSVM-UNet) approaches. For adjacent organs with similar intensities (kidneys adjacent to liver), the model successfully distinguishes organ boundaries where other methods produce segmentation leakage.

**Failure cases (Figure 5):**
1. **Small central aorta**: model partially identifies the aorta but misses the small central region in specific slices — sensitivity threshold for extremely small structures
2. **Pancreas in complex retroperitoneal environment**: model entirely fails to segment pancreas when retroperitoneal fat and gastrointestinal tissue present high intensity similarity — highly irregular morphology and low contrast remain challenging

---

## 8. Discussion

**Broader applicability:**
- **3D medical imaging**: LGF-VSS blocks can be extended by replacing 2D scanning with **3D tri-plane scanning** to capture volumetric context
- **Weakly supervised learning**: adaptive gating weights could serve as attention maps to guide segmentation under sparse annotations (scribbles or points)
- **Multi-modal fusion**: MCFB's dual-attention mechanism is naturally suited for CT and MRI registration, where it can align disparate feature spaces

**Known limitation**: framework is currently 2D-only. Future work will focus on scaling to large-scale 3D datasets and exploring efficacy in real-time surgical navigation.

---

## 9. Conclusion

LGFVM-UNet proposes three key innovations:
1. **LGF-VSS block**: integrates multi-directional selective scanning with parallel multi-scale convolutional branches through a QuadGate dynamic fusion mechanism
2. **MCFB**: establishes comprehensive bidirectional feature interactions between encoder and decoder hierarchies
3. **Gradient Statistics-based Adaptive Hierarchical Loss**: dynamically adjusts multi-level supervision weights based on real-time gradient statistics

Achieves **88.74% DSC and 6.65mm HD95** on Synapse — surpassing MSVM-UNet by +1.33% DSC and -2.26mm HD95 with marginal computational overhead (+0.13G FLOPs).

---

## References (Selected)

- Ronneberger et al., 2015 — U-Net
- Zhou et al., 2018 — UNet++
- Cao et al., 2022 — SwinUNet
- Chen et al., 2021 — TransUNet
- Ma and Wang, 2024 — Semi-Mamba-UNet
- Zhang et al., 2024 — VM-UNetV2
- Wu et al., 2025 — H-vmunet
- **Chen et al., 2024 — MSVM-UNet** ← direct comparison
- Ruan and Xiang, 2024 — VM-UNet
- Liu et al., 2024b — Swin-UMamba
- Gu and Dao, 2023 — Mamba
- Vanaja and Prakasam, 2025 — CBAM-AG UNet
- Jiang et al., 2025 — RWKV-UNet
