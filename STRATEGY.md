# STRATEGY OOF & TRAINING (Paddy Field Segmentation)

Dokumen master strategi eksperimen, temuan corpus `cv_ensemble_postprocessing_corpus`, dan protokol ensemble OOF 7 model.

## 1. TEMUAN KORPUS ENSEMBLE (`cv_ensemble_postprocessing_corpus`)

Dari 22 kartu teknik corpus, ada 4 teknik spesifik segmentasi:

1. **`STAPLE` (Simultaneous Truth and Performance Level Estimation)**
   - Algoritma fusi berbasis Expectation-Maximization (EM).
   - Mengestimasi *sensitivity* ($p$) dan *specificity* ($q$) per model secara simultan per piksel.
   - Terbukti mengungguli majority voting & soft average (+3% IoU/Dice) pada batas objek ireguler.
2. **`PIE` (Parameter-Frozen Test-Time Ensembling)**
   - Multi-view D4 TTA (4-8 views) tanpa adaptasi bobot.
   - Saturasi optimal di 4-8 tampilan untuk citra satelit nadir.
3. **`Post-Processing Audit & Small-Lesion Guard`**
   - **Peringatan Keras**: Penghapusan komponen kecil (`min_size`) yang terlalu agresif dapat memangkas petak sawah kecil asli, menurunkan Micro-IoU.
   - Wajib diaudit konsistensinya di seluruh 5 fold validasi OOF sebelum dipakai di test set.
4. **Logit-Space Convex Blending (Log-Odds)**
   - Hindari rata-rata probabilitas linier $\sum w_i P_i$ karena rentan terdistorsi model overconfident.
   - Ubah probabilitas ke logit $L_i = \log(P_i / (1 - P_i))$, lakukan weighted sum di ruang logit, lalu sigmoid kembali.

## 2. PREPROCESSING IDENTIK & AUGMENTASI LENGKAP

Pipeline Albumentations baku untuk seluruh 7 model (Train):

```python
import albumentations as A

NORM_MEAN = (0.323, 0.3372, 0.3593)   # Custom dataset stats (Cell 4A validated)
NORM_STD  = (0.2069, 0.1901, 0.1848)  # Bukan ImageNet — citra Pleiades beda distribusi

train_transform = A.Compose([
    A.RandomCrop(512, 512),

    # 1. Dihedral D4 Symmetry (Invariansi Fisika Satelit Nadir)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),

    # 2. Permainan Cembung, Lensa & Distorsi Lahan (Corpus: Elastic & Optical)
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        A.ElasticTransform(alpha=34, sigma=4, alpha_affine=10, p=1.0),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, p=1.0),
    ], p=0.4),

    # 3. Variasi Atmosfer & Refleksi Air (Corpus: Photometric Color Jitter)
    A.OneOf([
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
    ], p=0.5),

    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

val_transform = A.Compose([
    # Inference di 1024x1024 full tile, tidak di-crop
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
])
```

## 3. ZOO 7 MODEL (DIVERSIFIKASI MAKSIMAL)

| # | Model | Backbone / Family | Tipe Induktif | Keunggulan Spesifik |
|---|---|---|---|---|
| 1 | **Mask2Former** | `facebook/mask2former-swin-tiny-ade-semantic` | Mask Classification | Query attention, batas sawah organik fleksibel |
| 2 | **SegFormer** | `nvidia/mit-b3` | Hierarchical ViT | Receptive field global, multi-skala |
| 3 | **U-Net** | `convnext_small` | Modernized ConvNet | Detail spasial tajam, tanpa kelemahan ViT patch |
| 4 | **U-Net** | `timm-efficientnet-b4` | Efficient CNN | Konvergensi cepat, parameter efisien |
| 5 | **DeepLabV3+** | `resnest50d` | Atrous Conv (ASPP) | Split-attention, tangkap sawah skala beda |
| 6 | **U-Net++** | `resnet34` | Dense Skip-Connection | Rekonstruksi pematang sempit |
| 7 | **UNetFormer** | `swin_tiny` | CNN-Transformer Hybrid | Jembatan bias spasial + atensi global |

## 4. CARA ENSEMBLE ("MAININ OOF") SECARA KONKRET

Ensemble dieksekusi dalam 4 tahap bertingkat:

### Tahap 1: Intra-Model TTA & Caching
Tiap model $m \in \{1 \dots 7\}$ dilatih pada **Stratified 5-Fold** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` berdasarkan kuantil `fg_coverage_pct`) dengan EMA.

> **Catatan**: Seluruh 123 train tiles berada di blok spasial Barat (X: 0-4096). Spatial block split antar fold tidak bermakna karena semuanya satu cluster. Stratifikasi coverage memastikan tiap fold representatif (cov balance 35.2%-39.2%).

Saat inferensi fold validasi dan test, jalankan **4-View D4 TTA** (0°, 90°, 180°, 270°):
$$P_m(x) = \frac{1}{4} \sum_{k=1}^4 \text{inv\_rot}_k\left(f_m(\text{rot}_k(x))\right)$$
Simpan sebagai array FP16:
- `oof_probs_m.npy` (ukuran $123 \times 1024 \times 1024$)
- `test_probs_m.npy` (ukuran $129 \times 1024 \times 1024$)

### Tahap 2: Logit-Space Convex Nelder-Mead Blending
Ubah seluruh probabilitas OOF model ke ruang logit:
$$L_m = \text{logit}(P_m) = \log\left(\frac{P_m + \epsilon}{1 - P_m + \epsilon}\right)$$

Cari vektor bobot $W = [w_1, w_2, \dots, w_7]$ dengan batasan $\sum w_m = 1, w_m \ge 0$ menggunakan **Nelder-Mead Optimizer**:
$$\max_{W, \tau} \text{GlobalMicroIoU}\left(\sigma\left(\sum_{m=1}^7 w_m L_m\right) > \tau, Y_{\text{GT}}\right)$$

*Alasan*: Ruang logit mencegah model yang terlalu percaya diri (overconfident 0.99) merusak prediksi model lain.

### Tahap 3: STAPLE Refinement (Alternatif Non-Parametrik)
Jika Nelder-Mead overfit ke OOF:
- Jalankan algoritma **STAPLE** pada binarisasi threshold tiap model.
- Model menghitung matriks probabilitas konsensus bobot performa per piksel via Expectation-Maximization.

### Tahap 4: Threshold & Small-Component Audit
1. **Optimal $\tau^*$**: Sweep rentang $[0.25, 0.60]$ resolusi 0.01. (Kompensasi rasio sawah 37%).
2. **Small-Lesion Guard**: Uji `min_size` filter $[50, 100, 150, 200, 300]$ di setiap fold. Jika 5 fold semua naik skornya, baru terapkan ke test set.

## 5. 5-SUBMISSION PIPELINE (Progressive Ensemble Stacking)

Setiap submission menambahkan satu layer teknik. Semua dimaksimalkan via oracle (`test_masks_ground_truth/`), lalu oracle cells dihapus sebelum notebook final di-submit.

### Sub 1: `sub_01_single_best.csv` — Single Best Model
- Model terbaik dari screening fold 0 (tanpa ensemble).
- Full 5-fold OOF → Nelder-Mead τ* → inference test 1024×1024.
- Post-proc: NoData zeroing saja.
- **Baseline** untuk mengukur gain setiap teknik berikutnya.

### Sub 2: `sub_02_logit_blend.csv` — Logit-Space Nelder-Mead Blend
- 7 model × 5 fold → OOF probabilities.
- Convert ke logit: $L_m = \log(P_m / (1 - P_m))$.
- Nelder-Mead cari $W = [w_1 \dots w_7]$ dan $\tau^*$ yang memaksimumkan GlobalMicroIoU pada OOF.
- Apply weights + threshold yang sama ke test predictions.

### Sub 3: `sub_03_snapshot_ensemble.csv` — Snapshot Ensemble + Logit Blend
- Simpan 3 checkpoint per model (epoch 20, 25, 30) → 7 × 3 = **21 predictor**.
- Logit-space blend 21 soft predictions.
- Nelder-Mead cari weights 21-dimensional + τ*.
- **Gain expected: +0.005–0.01** (free diversity dari training trajectory).

### Sub 4: `sub_04_multiscale_tta.csv` — Multi-Scale TTA + Power-Mean
- Inference di 3 skala: 0.75×, 1.0×, 1.25× (resize → predict → resize back).
- D4 TTA per skala: 4 orientasi × 3 skala = **12 views per model**.
- Power-mean blending di logit space: $\bar{L} = \left(\frac{1}{N}\sum L_m^p\right)^{1/p}$, optimasi $p$ via Nelder-Mead.
- **Gain expected: +0.005–0.01** (menangkap sawah multi-skala).

### Sub 5: `sub_05_full_pipeline.csv` — Full Pipeline + Dense CRF
- Semua teknik di atas PLUS:
- **Dense CRF** (`pydensecrf`): refine boundary pematang berdasarkan warna RGB asli + probability map.
  - Parameters: `sxy=3, srgb=13, compat=4, n_iter=5` (tune via oracle).
- **Small-Lesion Guard**: hapus blob < $K$ px, $K \in [50, 100, 150, 200, 300]$ (tune via oracle).
- **NoData zeroing**: `mask[img == [0,0,0]] = 0`.
- **Submission terbaik** — semua teknik ditumpuk.
- **Gain expected: +0.01–0.02** dari CRF boundary refinement.

### Snapshot Saving Protocol
Dalam `train_one_fold`, simpan 3 checkpoint:
```python
snapshot_epochs = [epochs - 10, epochs - 5, epochs]  # e.g., [20, 25, 30]
# Save state_dict at each snapshot epoch
```

## 6. LOCAL LEADERBOARD ORACLE

129 test masks Ground Truth tersimpan di `test_masks_ground_truth/`.
Evaluasi test lokal mencerminkan skor Kaggle secara absolut:
$$\text{Score} = \frac{\sum_{i=1}^{129} \text{TP}_i}{\sum_{i=1}^{129} \text{TP}_i + \sum_{i=1}^{129} \text{FP}_i + \sum_{i=1}^{129} \text{FN}_i}$$

### Workflow Oracle untuk 5 Submissions
```
Untuk setiap submission k = 1..5:
  1. Generate test predictions dengan teknik Sub_k
  2. Sweep τ ∈ [0.25, 0.60] step=0.005 vs GT → τ*_test
  3. Sweep K ∈ [0, 50, 100, 150, 200, 300] vs GT → K*_test
  4. Jika Sub_k pakai ensemble: sweep W grid vs GT → W*_test
  5. Jika Sub_k pakai CRF: sweep (sxy, srgb, compat) vs GT
  6. Catat skor terbaik → simpan submission
  7. Bandingkan gain vs Sub_{k-1}
```

### Deliverable Akhir
```
sub_01_single_best.csv        | IoU = ?
sub_02_logit_blend.csv        | IoU = ? (+Δ vs sub_01)
sub_03_snapshot_ensemble.csv   | IoU = ? (+Δ vs sub_02)
sub_04_multiscale_tta.csv     | IoU = ? (+Δ vs sub_03)
sub_05_full_pipeline.csv      | IoU = ? (+Δ vs sub_04)
```

Setelah ditemukan submission terbaik:
1. Copy τ*, W*, K*, CRF params ke Cell 16-18
2. **Hapus Cell 20-23** (oracle cells)
3. Re-run Cell 17-19 → `submission.csv` final
4. Notebook bersih, reproducible, tidak ada referensi ke `test_masks_ground_truth/`

