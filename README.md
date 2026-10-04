<div align="center">
  <img src="logo.png" alt="Logo" width="128" height="128">
  
  <h1>paddy-field-segmentation-finals</h1>
  <p><strong>Pleiades High-Resolution Satellite Semantic Segmentation for National Food Security Mapping</strong></p>
  
  <p align="center">
    <img src="https://img.shields.io/badge/Competition-Holomine_AI_Finals-blue?style=flat-square" alt="Competition">
    <img src="https://img.shields.io/badge/Standing-Rank_2_National-success?style=flat-square" alt="Standing">
    <img src="https://img.shields.io/badge/Private_Score-0.68965-orange?style=flat-square" alt="Private Score">
    <img src="https://img.shields.io/badge/Public_Score-0.78169-brightgreen?style=flat-square" alt="Public Score">
    <img src="https://img.shields.io/badge/Language-Python_3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Language">
  </p>
  
  <p align="center">
    Heterogeneous deep learning ensemble solution combining CNNs, Feature Pyramid Atrous networks, and Vision Transformers for precise agricultural parcel delineation on 0.5-meter optical satellite imagery.
  </p>
</div>

## Overview

This repository contains the complete, leak-free, and reproducible winning solution for the Holomine Paddy Field Segmentation Finals. The task requires semantic binary segmentation of agricultural paddy fields across 1024x1024 optical RGB tiles captured by the Pleiades constellation over the Tangerang Sepatan Timur district.

The pipeline achieved Rank 2 National (Runner-Up) with a final Private Leaderboard score of 0.68965, demonstrating a climb of +5 positions from the public standings.

## Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## Core Architectural Pillars

1. Stratified 5-Fold Cross-Validation: Partitions 123 training tiles into balanced splits based on coverage quantiles and NoData proportion, preventing spatial distribution bias.
2. Heterogeneous 4-Family Model Ensemble: Combines four complementary vision architectures:
   * U-Net with SE-ResNeXt50 (channel-wise feature recalibration)
   * U-Net with EfficientNet-B4 (efficient compound scaling)
   * DeepLabV3+ with ResNeXt50 (atrous spatial pyramid pooling for multi-scale context)
   * MAnet with MiT-B3 (multi-scale position and channel attention transformer)
3. Logit-Space Convex Optimization: Model blend weights and global decision threshold are optimized jointly on Out-of-Fold (OOF) cross-validation logits using Nelder-Mead simplex search, strictly guarding against test set leakage.
4. Domain-Grounded Post-Processing: Enforces deterministic hard-zeroing on black orbit boundary pixels (`[0, 0, 0]`) and applies coverage-adaptive thresholding to prevent boundary under-prediction.

## Evaluation Metric

The competition evaluates submissions using the threshold-based Airbus Object Segmentation Beta metric. Per-image Intersection over Union (IoU) is calculated across 10 evaluation thresholds:

$$
T = \{0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95\}
$$

For each test image and threshold, a binary score is awarded if the IoU exceeds the threshold:

$$
\text{Score} = \frac{1}{N \cdot |T|} \sum_{i=1}^N \sum_{t \in T} \mathbb{I}(\text{IoU}_i > t)
$$

## Ablation Study

Progressive Out-of-Fold (OOF) validation performance across iterative engineering stages:

| Pipeline Stage | OOF Micro-IoU | Marginal Gain | Key Engineering Mechanism |
|---|:---:|:---:|---|
| Baseline Single Model (Fold 0) | 0.8822 | Reference | Single U-Net EfficientNet-B4 |
| Full 5-Fold Cross-Validation | 0.9103 | +0.0281 | Regional geographic variance reduction |
| Logit-Space Convex Blend | 0.9185 | +0.0082 | Multi-backbone feature complementary fusion |
| D4 Test-Time Augmentation | 0.9240 | +0.0055 | Rotation-invariant satellite symmetry exploitation |
| NoData Hard Zeroing | 0.9275 | +0.0035 | Eliminates swath boundary interpolation artifacts |

## Repository Structure

```
paddy-field-segmentation-finals/
├── rico sakit perut vs 100 gorila_Notebook_Babak Final.ipynb
├── submission.csv
├── logo.png
├── LICENSE
├── README.md
├── notebook-images/
└── holomine-paddy-field-segmentation-finals/
```

* `rico sakit perut vs 100 gorila_Notebook_Babak Final.ipynb`: The complete documented notebook with 25 narrative Markdown cells, 23 executed code cells, and embedded visualization plots.
* `submission.csv`: Final verified test set submission file containing column-major RLE masks for all 129 test tiles.
* `notebook-images/`: Extracted diagnostic figures including sensitivity curves, ablation charts, and model uncertainty maps.

## Reproducibility Protocol

All experiments are conducted with fixed deterministic seeds (`SEED = 42`) across PyTorch, CUDA, NumPy, and Python standard libraries.

1. Dependencies: Install packages via `pip install segmentation-models-pytorch albumentations timm transformers`.
2. Fold Partition: Stratified CV metadata is loaded directly from `train_folds_stratified.csv`.
3. Checkpoint Caching: Pre-computed model checkpoints and OOF arrays are referenced deterministically, allowing rapid verification without redundant re-training.