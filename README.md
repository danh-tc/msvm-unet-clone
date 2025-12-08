<div align="center">
<h1>MSVM-UNet: Multi-Scale Vision Mamba UNet for Medical Image Segmentation</h1>

[Chaowei Chen](mailto:chishengchen@stu.ynu.edu.cn)<sup>1</sup>,[Li Yu](mailto:yuli0501@163.com)<sup>
1</sup>,[Shiquan Min](mailto:minshiquan@mail.ynu.edu.cn)<sup>1</sup>, [Shunfang Wang](mailto:sfwang_66@ynu.edu.cn)<sup>
1,2,*</sup>

<div><sup>1</sup>School of Information Science and Engineering, Yunnan University, Kunming, 650504, Yunnan, China</div>
<div><sup>2</sup>Yunnan Key Laboratory of Intelligent Systems and Computing, Yunnan University, Kunming, 650504, Yunnan, China</div>

Paper: ([arXiv 2408.13735](https://arxiv.org/abs/2408.13735))

</div>

## Abstract

State Space Models (SSMs), especially Mamba, have shown great promise in medical image segmentation due to their ability
to model long-range dependencies with linear computational complexity. However, accurate medical image segmentation
requires the effective learning of both multi-scale detailed feature representations and global contextual dependencies.
Although existing works have attempted to address this issue by integrating CNNs and SSMs to leverage their respective
strengths, they have not designed specialized modules to effectively capture multi-scale feature representations, nor
have they adequately addressed the directional sensitivity problem when applying Mamba to 2D image data. To overcome
these limitations, we propose a Multi-Scale Vision Mamba UNet model for medical image segmentation, termed MSVM-UNet.
Specifically, by introducing multi-scale convolutions in the VSS blocks, we can more effectively capture and aggregate
multi-scale feature representations from the hierarchical features of the VMamba encoder and better handle 2D visual
data. Additionally, the large kernel patch expanding (LKPE) layers achieve more efficient upsampling of feature maps by
simultaneously integrating spatial and channel information. Extensive experiments on the Synapse and ACDC datasets
demonstrate that our approach is more effective than some state-of-the-art methods in capturing and aggregating
multi-scale feature representations and modeling long-range dependencies between pixels.

## Overview

<img src="./assets/overall.png" alt="overall"  />

## Main Results

- Synapse Multi-Organ Segmentation

![image-20240825134505994](./assets/image-20240825134505994.png)

- ACDC for Automated Cardiac Segmentation

![image-20240825134539739](./assets/image-20240825134539739.png)

## Installation

We recommend the following platforms:

```
Ubuntu <= 22 / CUDA 11.8.0 / Python 3.8 / Pytorch >= 2.0.0
```

### Step 1: Install CUDA 11.8.0 (Skip if already installed)

Based on your environment, you can install CUDA 11.8.0 in your home directory. This step downloads
the [CUDA toolkit installer](https://developer.nvidia.com/cuda-11-8-0-download-archive), makes it executable, and
installs it to `$HOME/cuda-11.8` directory:

```bash
chmod +x cuda_11.8.0_520.61.05_linux.run
./cuda_11.8.0_520.61.05_linux.run --silent --toolkit --override --installpath=$HOME/cuda-11.8
```

### Step 2: Configure CUDA Environment Variables

Add CUDA to your system PATH and library path by editing `~/.bashrc`. These commands set up the environment variables so
your system can find the CUDA compiler and libraries:

```bash
vim ~/.bashrc
# Add the following lines to ~/.bashrc:
export CUDA_HOME=$HOME/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# Then reload the configuration:
source ~/.bashrc
```

### Step 3: Verify CUDA Installation

Check that CUDA 11.8 is correctly installed by running the NVIDIA CUDA compiler version command. You should see output
similar to:

```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

### Step 4: Create Conda Environment and Install PyTorch

Create a new conda environment named `msvmunet` with Python 3.8, activate it, and install PyTorch 2.0.0 with CUDA 11.8
support:

```bash
conda create -n msvmunet python=3.8
conda activate msvmunet
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 5: Check and Install GCC 11

Check if your GCC version is 11. If not, install GCC 11 and G++ 11. The Selective Scan CUDA kernels require GCC 11 for
proper compilation:

```bash
$ gcc --version
gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

If your GCC version is not 11, install it and set it as the default compiler:

```bash
sudo apt install gcc-11 g++-11

vim ~/.bashrc
# Add the following lines to ~/.bashrc:
export CC=gcc-11
export CXX=g++-11
# Then reload the configuration:
source ~/.bashrc
```

### Step 6: Install Selective Scan Module

Install the Triton-implemented Selective Scan module, which is a core component of the Mamba architecture. This step
compiles the CUDA kernels using GCC 11:

```bash
cd kernels/selective_scan
CC=gcc-11 CXX=g++-11 pip install -e .
```

### Step 7: Install Python Dependencies

Install all required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 8: Install medpy (Optional but Recommended)

The `medpy` package includes `SimpleITK`, which sometimes fails to install automatically because it needs to download
from GitHub. To avoid potential installation issues, install `medpy` directly without installing its dependencies (other
required dependencies are already covered in `requirements.txt`):

```bash
pip install medpy==0.5.2 --no-deps
```

### Step 9: Install Additional Dependencies for Baseline Models (Optional)

If you need to reproduce baseline methods such as VM-UNet or Swin-UMamba, install the following additional dependencies:

```bash
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```

If the above installation gets stuck or fails, you can download the offline wheel packages and install them
locally: [causal_conv1d](https://drive.google.com/file/d/1blBHfl2UIK5GjaFm0wXY-vIWJ6arUp5a/view?usp=drive_link), [mamba_ssm](https://drive.google.com/file/d/1D7uRpEBzZ-UoYeQxpZB9z8lFvHpZrlmN/view?usp=drive_link).

```bash
pip install causal_conv1d-1.2.0.post2+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm-1.0.1+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

### Note for Cloud Environments

If you are using a cloud computing platform such as [GPUShare](https://gpushare.com), simply select a machine that
matches our recommended environment specifications. In this case, you can skip **Steps 1, 2, 3, and 5** as CUDA and GCC
should already be properly configured on the cloud instance.

## Environment Variables Setup

Before preparing data and training models, you need to set up the following environment variables in your shell configuration file (`~/.bashrc` or `~/.zshrc`):

```bash
# Add these lines to ~/.bashrc or ~/.zshrc
export DATASET_HOME=/path/to/your/datasets    # Root directory for all datasets
export PRETRAIN_HOME=/path/to/your/pretrained # Directory for pretrained model weights

# Then reload the configuration:
source ~/.bashrc  # or source ~/.zshrc
```

**Example Setup:**
```bash
export DATASET_HOME=$HOME/datasets
export PRETRAIN_HOME=$HOME/pretrained_models
```

These environment variables are used by the training and testing scripts to locate datasets and pretrained models, making it easier to manage different data locations across different machines.

## Prepare Data & Pretrained Model

### Dataset Preparation

#### 1. Create Dataset Directory Structure

```bash
mkdir -p $DATASET_HOME/mis
```

#### 2. Download and Setup Synapse Multi-Organ Dataset

- **Option 1:** Sign up at the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480) and download the dataset
- **Option 2:** Download the [preprocessed data](https://drive.google.com/file/d/1BvpY0g9mKkkhdHpAX1HqDw8iTJNbFuwq/view?usp=drive_link)

Extract the dataset to: `$DATASET_HOME/mis/synapse/`

#### 3. Download and Setup ACDC Dataset

Download the preprocessed ACDC dataset from [Google Drive of MT-UNet](https://drive.google.com/file/d/1fiGIevmbfLwvHUblYaDhDdC0aHlt277f/view?usp=drive_link)

Extract the dataset to: `$DATASET_HOME/mis/acdc/`

**Expected Directory Structure:**
```
$DATASET_HOME/
└── mis/
    ├── synapse/
    │   ├── train/
    │   └── test/
    └── acdc_01/
        ├── train/
        ├── valid/
        └── test/
```

### Pretrained Model Setup

#### 1. Create Pretrained Model Directory

```bash
mkdir -p $PRETRAIN_HOME
```

#### 2. Download VMamba-Tiny V2 Pretrained Weights

Download the pretrained VMamba-Tiny V2 model from VMamba official release:
- **Model:** [vssm1_tiny_0230s_ckpt_epoch_264.pth](https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth)
- **Save to:** `$PRETRAIN_HOME/vssm1_tiny_0230s_ckpt_epoch_264.pth`

This pretrained model is used for encoder initialization to improve training convergence and performance.

**Expected Directory Structure:**
```
$PRETRAIN_HOME/
└── vssm1_tiny_0230s_ckpt_epoch_264.pth
```

## Training

Using the following command to train & evaluate MSVM-UNet:

```bash
bash ./run_msvm_unet.sh
```

**Note:** The `model/` directory contains implementations of several baseline methods (Att-UNet, Trans-UNet, Swin-UNet, VM-UNet, Swin-UMamba, etc.) for comparison purposes. These are not required for MSVM-UNet training but are included for reproducibility of the experiments in the paper.

## Citation

```
@article{chen2024msvmunet,
  title={MSVM-UNet: Multi-Scale Vision Mamba UNet for Medical Image Segmentation}, 
  author={Chaowei Chen and Li Yu and Shiquan Min and Shunfang Wang},
  journal={arXiv preprint arXiv:2408.13735},
  year={2024}
}
```

## Acknowledgements

We thank the authors
of [TransUNet](https://github.com/Beckschen/TransUNet), [SLDGroup](https://github.com/SLDGroup), [Mamba](https://github.com/state-spaces/mamba), [VMamba](https://github.com/MzeroMiko/VMamba), [VM-UNet](https://github.com/JCruan519/VM-UNet),
and [Swin-UMamba](https://github.com/JiarunLiu/Swin-UMamba) for making their valuable code & data publicly available.
