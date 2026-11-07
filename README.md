# Dynamic Alignment and Prototype-Guided Enhancement Network for Small Object Detection

This repository contains the implementation of **DAPENet**, a small object detection framework for aerial images. DAPENet improves multi-scale feature fusion by addressing spatial misalignment and recovering fine-grained object details.

> **Paper:** *Dynamic Alignment and Prototype-Guided Enhancement Network for Small Object Detection*  
> **Framework:** MMDetection  
> **Base detector:** Generalized Focal Loss (GFL)

## Overview

Small objects in aerial images occupy only a few pixels and are easily obscured by background clutter. Repeated downsampling also shifts the spatial representation of objects across feature levels. DAPENet addresses these issues with two complementary components:

- **Dynamic Deformable Alignment (DDA)** estimates offsets and modulation masks to align adjacent feature levels before fusion. It suppresses low-contribution sampling points to reduce background interference.
- **Prototype-Guided Feature Enhancement (PGFE)** extracts object details from high-resolution features, stores them in a prototype bank, and selectively injects matched details into fused pyramid features using a similarity-based gate.

The framework uses a backbone to produce multi-scale features, applies DDA before cross-level fusion, enhances fused features with PGFE, and sends them to the detection head.

![fig](assets/frameworkfinal.pdf)

## Method

### Dynamic Deformable Alignment (DDA)

For each adjacent pair of feature levels, DDA upsamples the higher-level feature and concatenates it with the lower-level feature. A sampling-point estimation block predicts deformable offsets and modulation masks. Candidate locations with mask values below the threshold are suppressed, and the remaining samples are used to deformably align the higher-level feature before it is fused with the lower-level feature.

The paper uses a 3 × 3 sampling grid (`K = 9`), one deformable group, and a mask threshold of `0.3`.

### Prototype-Guided Feature Enhancement (PGFE)

PGFE extracts object-detail descriptors from the high-resolution feature map using predicted boxes, RoIAlign, and global average pooling. Descriptors update a shared prototype bank. Spatial queries from the fused pyramid features retrieve their most similar prototypes; a learnable sigmoid gate controls how much retrieved detail is added. PGFE is enabled after a warm-up period so that box predictions are sufficiently stable.

<!-- Add DDA/PGFE module diagrams from the paper here if available. -->

## Results

The paper evaluates DAPENet on VisDrone and UAVDT. Reported mAP results are summarized below.

| Dataset | Detector / Backbone | mAP | mAP50 | mAP75 |
|---|---|---:|---:|---:|
| VisDrone | DAPENet, ResNet-18 | 27.5 | 47.1 | 27.4 |
| VisDrone | DAPENet, ResNet-50 | 30.1 | 50.6 | 30.6 |
| UAVDT | DAPENet | 17.3 | 30.5 | 18.5 |

On VisDrone, the paper reports 27.5 mAP with ResNet-18, compared with 25.9 for the GFL baseline. On UAVDT, DAPENet reports 17.3 mAP, compared with 14.0 for GFL. The paper also reports 169.3 GFLOPs and 21.4M parameters for its VisDrone ResNet-18 setting.

<!-- Optional: add qualitative comparison figures from the paper here. -->

## Requirements

The project environment follows the original repository setup:

- Python and CUDA versions should match the installed PyTorch build.
- PyTorch 1.10.1
- MMDetection 2.24.1
- MMCV-full 1.5.1
- MMDetection requirement: `mmdet >= 2.7.0`, `mmcv-full >= 1.4.2`
- Albumentations and other packages listed in `requirements/albu.txt`

Install the MMDetection toolbox and project dependencies according to the [MMDetection getting started guide](docs/en/get_started.md).

```bash
# Install PyTorch 1.10.1 using the command appropriate for your CUDA version.
# See https://pytorch.org/ for the matching installation command.

# Install the versions used by this project.
pip install mmcv-full==1.5.1
pip install mmdet==2.24.1

# Install additional dependencies.
pip install nltk
pip install -r requirements/albu.txt

# Build and install the sparse convolution extension.
cd Sparse_conv
python setup.py install
cd ..
```

## Data Preparation

Download and prepare the VisDrone and UAVDT datasets in COCO format. The original README points to the datasets' official sources and [UFPMP-Det](https://github.com/PuAnysh/UFPMP-Det) as an alternative source.

Update dataset paths in the corresponding configuration files under [`configs/UAV`](configs/UAV) before training or evaluation. Dataset directory names and annotation paths must match the configuration.

## Training

Train on one GPU:

```bash
python tools/train.py /path/to/config-file --work-dir /path/to/work-dir
```

Train on multiple GPUs:

```bash
bash tools/dist_train.sh /path/to/config-file num-gpus --work-dir /path/to/work-dir
```

The paper's implementation details specify GFL as the base detector, batch size 2, and SGD optimizer. For VisDrone, the reported training schedule is 15 epochs with input size 1333 × 800. For UAVDT, it is 6 epochs with input size 1024 × 540 and learning-rate drops after epochs 4 and 5. PGFE uses a warm-up: disabled in epochs 1–2 on VisDrone and epoch 1 on UAVDT. Refer to the actual config files for the exact settings used by this codebase.

## Evaluation

```bash
python tools/test.py /path/to/config-file /path/to/checkpoint.pth --eval bbox
```

For example, to evaluate the latest checkpoint in a work directory:

```bash
python tools/test.py /path/to/config-file /path/to/work-dir/latest.pth --eval bbox
```



The provided PDF is anonymized and contains placeholder DOI and ISBN fields. Replace the author field and add final publication metadata when those details are available.
