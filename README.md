# Evaluating the Effect of Multimodal MRI Channels on Automated 3D Glioma Segmentation using 3D U-Net

**COMP3931 Individual Project — University of Leeds, 2024/25 · Distinction**

Supervised by Dr. Arash Rabbani and Dr. Yongxing Wang

---

## Research Question

> Does increasing the number of MRI input modalities actually improve brain tumour segmentation, or does the literature's default assumption — "more channels = better" — need to be challenged?

Most brain tumour segmentation studies adopt all four available MRI modalities without systematically isolating the effect of channel count. This project treats channel count as a controlled variable across three 3D U-Net configurations, holding everything else fixed: architecture, loss function, optimiser, data split, and augmentation pipeline.

---

## Key Finding

**Modality coordination matters more than modality count.**

- Adding a third MRI channel (T2) made segmentation *worse*, not better — the Tri-modal configuration underperformed Di-modal on sub-region delineation despite receiving more information.
- Quad-modal achieved the best overall Dice (~0.52) and fastest convergence, but Di-modal produced better sub-region boundary delineation.
- The results challenge the assumption that stacking all available modalities is the optimal strategy for glioma segmentation.

---

## Model Configurations

| Configuration | Modalities | Dice (↑) | Overall Accuracy | Segmentation Accuracy |
|---|---|---|---|---|
| **Di-modal** | FLAIR + T1CE | — | 95.0% | 45.7% |
| **Tri-modal** | FLAIR + T1CE + T2 | ~0.44 | 94.8% | 30.6% ← worst |
| **Quad-modal** | FLAIR + T1CE + T2 + T1 | ~0.52 | 96.0% | 46.6% |

Segmentation accuracy is the average per-class accuracy across tumour sub-regions (necrotic, oedema, enhancing tumour), excluding the background class — the meaningful metric under severe class imbalance.

---

## Dataset

**BraTS-2020** via [Larxel's Kaggle repository](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

- 369 patient directories, each containing four MRI modalities (`flair`, `t1`, `t1ce`, `t2`) and a ground-truth segmentation (`seg`) in `.nii.gz` format.
- Original scan dimensions: 240 × 240 × 155 voxels. Cropped to 128 × 128 × 128 for training.
- Labels: 0 = background, 1 = necrotic/core, 2 = oedema, 4 = enhancing tumour.
- Class imbalance: ~95–98% of voxels are background — a major design constraint addressed by cropping and augmentation, but not fully resolved.

**Data split (70/20/10):** 257 train · 74 validation · 37 test patients

> Note: official BraTS challenge registration was closed at project start; the Larxel Kaggle mirror (usability score 8.24/10) was selected for its completeness and adherence to the official naming conventions.

---

## Architecture

A modified 3D U-Net based on [Ronneberger et al. (2015)](https://arxiv.org/pdf/1505.04597):

- **5 encoder/decoder levels** (deeper than the original)
- **Dropout (p=0.2)** at the bottleneck for regularisation
- **Transposed convolutions** for upsampling; skip connections from encoder to decoder
- **Number of input channels** is passed as a parameter — the only variable changed between configurations

**Training setup:**

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.0001 |
| Loss function | Categorical cross-entropy (4 classes) |
| Epochs | 20 (max) |
| Early stopping | Patience 5 (validation loss) |
| Batch size | Limited by Colab VRAM |
| Input shape | 128 × 128 × 128 |

**Metrics tracked:** Accuracy, MeanIoU, Dice (overall + per-class: necrotic, oedema, enhancing), Sensitivity, Precision, Specificity.

---

## Data Preprocessing & Augmentation

All preprocessing occurs on-the-fly in a custom data generator (avoids materialising the full augmented dataset, which would exceed Colab storage).

**Preprocessing:**
- Min-max intensity normalisation per scan: `x_norm = (x − x_min) / (x_max − x_min)`
- Cropping from 240×240×155 → 128×128×128 (centres the brain, removes background skull, ensures power-of-2 dimensions for the encoder/decoder)

**Augmentation (training set only, 10% probability each):**
- Additive Gaussian noise
- Random translations
- Random scaling
- Random flips and orientation changes

---

## Setup & Requirements

```bash
# Clone the repository
git clone https://github.com/rubayeaAlketbi/Multimodal-MRI-Channels-on-Automated-3D-Glioma-Segmentation-using-3D-U-Net.git
cd Multimodal-MRI-Channels-on-Automated-3D-Glioma-Segmentation-using-3D-U-Net
```

**Dependencies:**

```
Python 3.11
TensorFlow 2.x + Keras
nibabel
scikit-learn
matplotlib
numpy
kaggle (for dataset download)
```

```bash
pip install tensorflow nibabel scikit-learn matplotlib numpy kaggle
```

**Recommended environment:** Google Colab (GPU runtime). The project was developed on Colab due to CUDA/TensorFlow version conflicts on a local RTX 3050 Laptop GPU.

**Dataset download (requires a Kaggle API token):**

```bash
kaggle datasets download awsaf49/brats20-dataset-training-validation
unzip brats20-dataset-training-validation.zip
```

---

## Usage

Open the main notebook and run cells sequentially. The notebook is structured to:

1. Download and prepare the BraTS-2020 dataset
2. Run class-distribution analysis
3. Build and compile the 3D U-Net for a specified channel configuration
4. Train all three configurations (Di / Tri / Quad-modal) sequentially, logging metrics per epoch
5. Evaluate on the held-out test set (37 patients) with confusion matrices and Dice scores
6. Generate 3D qualitative segmentation renders using the jet colour scheme

To select a configuration, change the `modalities` dictionary before training:

```python
modalities = {
    'Di-Modal':   ['flair', 't1ce'],
    'Tri-Modal':  ['flair', 't1ce', 't2'],
    'Quad-Modal': ['flair', 't1ce', 't2', 't1'],
}
```

Trained models are saved as `.h5` files and can be reloaded for inference without re-training.

---

## Limitations

These are stated explicitly in the project report:

- **Class imbalance unresolved:** ~95–98% of voxels are background. Categorical cross-entropy allows background to dominate the gradient. Dice loss or focal loss would be a more appropriate objective for a future iteration.
- **Best Dice (~0.52) is well below BraTS SOTA (~0.85+):** by design — this was a controlled single-variable ablation on free Colab compute (20 epochs, 128³ crop, single architecture). It was not a leaderboard submission. Conflating the two would be a methodological error.
- **37 test patients:** the trend is suggestive, not definitive. A larger registered-data replication is needed before making clinical claims.
- **Not clinically validated:** even the best configuration localises the tumour but does not meet clinical accuracy thresholds. This is a research tool, not a deployable medical device.
- **Data provenance:** Kaggle mirror, not the canonical BraTS registration source. Acknowledged as a limitation.

---

## Ethical Considerations

The project report includes a full legal, social, ethical, and professional analysis:

- **Data privacy:** BraTS-2020 is anonymised and used for non-commercial, academic purposes — consistent with dataset licensing and GDPR/HIPAA principles for anonymised medical data.
- **Bias:** class imbalance introduces bias toward background detection; this degrades sub-region segmentation and could harm clinical decision-making if applied uncritically.
- **Transparency:** 3D U-Net is a black-box to clinical practitioners. Explainability is a prerequisite for clinical adoption and is not addressed in this project.
- **Role of AI:** the system is designed as a decision-support tool to assist, not replace, clinical specialists.

---

## Repository Structure

```
├── notebook/
│   └── glioma_segmentation.ipynb   # Main implementation notebook
├── report/
│   └── final_report.pdf            # Full project report (COMP3931)
└── README.md
```

---

## Acknowledgements

Supervised by **Dr. Arash Rabbani** and **Dr. Yongxing Wang**, School of Computing, University of Leeds.

Dataset: BraTS-2020 via the International Brain Tumour Segmentation Challenge (MICCAI/ASNR).

Architecture based on: Ronneberger, O., Fischer, P. and Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597).

---

## Citation

If you reference this project:

```
AlKetbi, R.H. (2025). Evaluating the Effect of Multimodal MRI Channels on Automated
3D Glioma Segmentation using 3D U-Net. BSc Individual Project (Distinction),
University of Leeds, COMP3931.
```
