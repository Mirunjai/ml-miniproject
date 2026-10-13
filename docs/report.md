# Project Report: Structural Damage Recognition Using Transfer Learning

**Course:** UE24CS352A — Machine Learning (Sec-C, Team C-22, Project 31)  
**Students:** Mirunjai Suresh Kumar (PES1UG24AM161), Mohammed Sinan M T (PES1UG24AM165)  
**Date:** October 2026  

---

## Abstract
Post-disaster structural integrity assessment is essential for life safety and emergency resource prioritization, but manual visual inspection by civil engineers is hazardous, slow, and labor-intensive. This project develops an automated computer vision pipeline for binary structural damage recognition (`Damaged` vs. `Undamaged`) on the benchmark PEER Hub ImageNet ($\Phi$-Net) Task 2 dataset. Given hardware constraints (dual-core CPU, 8 GB RAM), we formulate a memory-efficient transfer learning strategy using an ImageNet-pretrained MobileNetV2 backbone (`include_top=False`) with disk-backed bottleneck feature caching. Evaluated on the official held-out test set of 1,460 images, the model achieves **84.25% test accuracy**, an **ROC-AUC of 0.9289**, and a **Damaged class recall of 85.91%** ($F_1 = 0.85$). An interactive Streamlit web application provides real-time inference with engineering safety disclaimers.

---

## 1. Problem Statement & Objectives
Following seismic events and natural catastrophes, rapid categorization of building facades, joints, columns, and beams into damaged or undamaged states is critical for emergency management. The objectives of this project are:
1. Develop an image classification pipeline to distinguish damaged structural elements (`Class 0`) from sound structures (`Class 1`).
2. Implement a memory-mapped, disk-cached data pipeline that trains efficiently on consumer laptop hardware without out-of-memory crashes.
3. Validate performance on the official benchmark test set across accuracy, ROC-AUC, precision, recall, and confusion matrix metrics.
4. Deploy a functional web application for single-photograph inference.

---

## 2. Dataset & Preprocessing

### 2.1 Dataset Composition
We utilize the **PEER Hub ImageNet ($\Phi$-Net) Task 2 (Damage State)** dataset (Gao & Mosalam, 2018; 2020), consisting of structural photographs cropped to $224 \times 224 \times 3$:
* **Training Set:** 11,811 images (6,282 Damaged [53.19%], 5,529 Undamaged [46.81%]), occupying ~6.62 GiB on disk.
* **Official Test Set:** 1,460 images (745 Damaged [51.03%], 715 Undamaged [48.97%]), occupying ~838 MiB.
* **Internal Validation Split:** Stratified 80/20 split on training data (9,448 train / 2,363 validation; random seed 42).

### 2.2 Preprocessing & Channel Inversion
The dataset arrays were preprocessed with **Caffe-style BGR ImageNet mean subtraction** ($\mu_{BGR} = [103.939, 116.779, 123.68]$), resulting in pixel values from $-123.68$ to $151.06$. Our pipeline restores true RGB values via:
$$\text{RGB}_{[0, 255]} = \text{clip}(\text{Image}_{BGR} + \mu_{BGR}, 0, 255)_{[\dots, ::-1]}$$
For model input, images are normalized to $[-1.0, 1.0]$ via $x / 127.5 - 1.0$. Ordinary user photographs uploaded in the web application are already standard RGB in $[0, 255]$ and bypass BGR mean restoration, directly undergoing Lanczos resizing and $[-1, 1]$ normalization.

---

## 3. Methodology & System Architecture

### 3.1 Transfer Learning & Bottleneck Feature Caching
Training deep CNN architectures end-to-end on consumer dual-core CPUs causes thermal throttling and system freezing. We employ **transfer learning with a frozen backbone**:
* **Feature Extractor:** MobileNetV2 pretrained on ImageNet (`include_top=False`, 2,257,984 frozen non-trainable parameters in convolutional blocks).
* **Disk-Backed Memory Mapping:** Features are extracted in mini-batches of 8 and written directly to disk via `np.lib.format.open_memmap`, accompanied by an atomic `.meta.json` progress sidecar. This bounds active Python heap RAM below 1.5 GB.
* **Classification Head:** Global Average Pooling outputs a 1,280-dimensional feature vector, followed by a Dropout layer ($p = 0.4$), a Dense ReLU layer (64 units, 81,984 params), and a two-class Softmax layer (130 params), totaling 82,114 trainable parameters (2,340,098 total model parameters).

```
Input Image (224×224×3) 
    ──> MobileNetV2 Backbone (Frozen ImageNet Weights, 2,257,984 params)
    ──> GlobalAveragePooling2D (1,280-dim feature vector)
    ──> [Cached to disk via open_memmap]
    ──> Dropout (rate = 0.4) 
    ──> Dense (64 units, ReLU, 81,984 params) 
    ──> Dense (2 units, Softmax, 130 params) 
    ──> [Score(Damaged), Score(Undamaged)]
```

### 3.2 Optimization Setup
* Optimizer: Adam ($\text{lr} = 0.001$), Loss: Categorical Cross-Entropy, Batch Size: 64.
* Callbacks: EarlyStopping ($\text{patience} = 5$, best weights restored), ReduceLROnPlateau ($\text{factor} = 0.5$, $\text{patience} = 2$).

---

## 4. Experimental Results & Discussion

### 4.1 Quantitative Performance
The model converged in 19 epochs (EarlyStopping triggered). Evaluation on the **1,460 official test samples** yielded:

| Metric | Damaged (`Class 0`) | Undamaged (`Class 1`) | Macro Average | Overall / Weighted |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | **0.8366** (83.7%) | **0.8489** (84.9%) | 0.8428 (84.3%) | 0.8426 (84.3%) |
| **Recall** | **0.8591** (85.9%) | **0.8252** (82.5%) | 0.8421 (84.2%) | 0.8425 (84.2%) |
| **$F_1$-Score** | **0.8477** (84.8%) | **0.8369** (83.7%) | 0.8423 (84.2%) | 0.8424 (84.2%) |
| **Test Accuracy** | — | — | — | **84.25%** |
| **ROC-AUC (Damaged)** | — | — | — | **0.9289** |

### 4.2 Confusion Matrix Analysis
$$\begin{bmatrix} \text{Actual Damaged (745)} \\ \text{Actual Undamaged (715)} \end{bmatrix} \times \begin{bmatrix} \text{Pred Damaged} & \text{Pred Undamaged} \end{bmatrix} = \begin{bmatrix} \mathbf{640} & \mathbf{105} \\ \mathbf{125} & \mathbf{590} \end{bmatrix}$$

* **False Negatives ($FN = 105$):** 14.1% miss rate. Plausible visual failure modes include micro-hairline fissures smoothed by downsampling, severe directional shadows, or low-contrast surface defects.
* **False Positives ($FP = 125$):** 17.5% false alarm rate. Plausible visual failure modes include high-contrast architectural brick seams, deep mortar joints, and external conduits mimicking fracture lines.
* **Interpretation Scope:** These represent visual hypotheses identified during model diagnostic review rather than on-site physical engineering forensics.

### 4.3 Training Convergence
Training loss decreased smoothly from 0.52 to 0.22, with validation loss stabilizing at 0.34 and validation accuracy reaching 85.7%. The small generalization gap confirms effective regularization via Dropout ($p = 0.4$) and feature freezing.

---

## 5. Conclusions, Limitations & Future Work

### Conclusions
1. MobileNetV2 transfer learning with disk-backed feature caching provides an effective solution for structural damage recognition, achieving **84.25% accuracy** and **0.9289 ROC-AUC** while adhering to strict consumer laptop hardware bounds.
2. The Streamlit prototype enables responsive interactive inference with single-image upload support.

### Limitations & Ethical Considerations
* **Binary Granularity:** The model categorizes damage coarsely without assessing structural severity levels (minor, moderate, severe) or failure mechanisms (shear vs. flexure).
* **Safety Disclaimer:** Machine learning predictions are assistive diagnostic tools and must **not** replace licensed structural engineering certification.

### Future Work
* Multi-class classification across $\Phi$-Net damage sub-tasks (Tasks 3–6).
* Grad-CAM attention visualisations to pinpoint crack localization.
* Edge deployment on mobile devices for offline post-disaster field usage.

---

## References
1. Gao, Y., & Mosalam, K. M. (2018). *Deep Transfer Learning for Image‐Based Structural Damage Recognition.* Computer‐Aided Civil and Infrastructure Engineering, 33(9), 748–768.
2. Gao, Y., & Mosalam, K. M. (2020). *PEER Hub ImageNet: A Large-Scale Multiattribute Benchmark Data Set of Structural Images.* Journal of Structural Engineering, 146(10), 04020198.
3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018, 4510–4520.
4. Ho, M., & Troncoso, J. (2018). *Structural Damage Image Classification.* Stanford University CS229 Machine Learning Project Report.
