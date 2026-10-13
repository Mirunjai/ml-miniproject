# Structural Damage Recognition

**UE24CS352A — Machine Learning Mini-Project**  
**Section:** Sec-C &nbsp;|&nbsp; **Team ID:** C-22 (Team 22) &nbsp;|&nbsp; **Project Number:** 31  
**Problem Statement:** Structural Damage Recognition  
**Dataset:** PEER Hub ImageNet ($\Phi$-Net), Task 2: Damage State  
**Primary Architecture:** MobileNetV2 Transfer Learning with Disk-Backed Bottleneck Feature Caching  

---

## 1. Project Overview & Problem Statement

Rapid, post-disaster structural assessment is critical for life safety, urban planning, and emergency relief distribution following earthquakes and natural disasters. Traditional building inspection involves physical on-site evaluation by licensed civil and structural engineers, which is hazardous, labor-intensive, and inherently slow during large-scale urban catastrophes.

This project implements an end-to-end Machine Learning pipeline that takes single structural surface photographs and classifies them into:
* **`0: Damaged` (D)**: Exhibiting structural cracks, spalling, exposed rebar, column failure, or shear distress.
* **`1: Undamaged` (UD)**: Exhibiting intact, sound structural conditions.

The deliverable includes an out-of-memory safe CPU training pipeline tailored to consumer hardware (Intel Core i3, 8 GB RAM), rigorous evaluation on the benchmark held-out test set (1,460 images), and a responsive **Streamlit** web application for real-time inference.

---

## 2. Team Members & Project Metadata

| Role | Student Name | SRN | Section & Team ID |
| :--- | :--- | :--- | :--- |
| **Student 1** | **MIRUNJAI SURESH KUMAR** | `PES1UG24AM161` | Sec-C, Team C-22 |
| **Student 2** | **MOHAMMED SINAN M T** | `PES1UG24AM165` | Sec-C, Team C-22 |

* **Course:** UE24CS352A — Machine Learning
* **Department:** Department of Computer Science and Engineering
* **Evaluation Date:** October 2026

---

## 3. Dataset Information ($\Phi$-Net Task 2)

The dataset is drawn from the **PEER Hub ImageNet ($\Phi$-Net)** benchmark developed by the Pacific Earthquake Engineering Research Center (UC Berkeley):

* **Source**: Yuqing Gao & Khalid M. Mosalam (UC Berkeley, 2018; 2020)
* **License**: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
* **Image Dimensions**: $224 \times 224 \times 3$
* **Storage Format**: Stored as `.npy` floating-point arrays

### Sample Counts and Class Balance

| Split | Damaged (Class 0) | Undamaged (Class 1) | Total Samples | Array Size on Disk |
| :--- | :---: | :---: | :---: | :---: |
| **Full Training Set** (`task2_X_train.npy`, `task2_y_train.npy`) | 6,282 (53.19%) | 5,529 (46.81%) | **11,811** | ~6.62 GiB |
| **Training Subset (80% Stratified)** | 5,026 (53.20%) | 4,422 (46.80%) | **9,448** | (Subset of train) |
| **Validation Subset (20% Stratified)** | 1,256 (53.15%) | 1,107 (46.85%) | **2,363** | (Subset of train) |
| **Official Held-Out Test Set** (`task2_X_test.npy`, `task2_y_test.npy`) | 745 (51.03%) | 715 (48.97%) | **1,460** | ~838.36 MiB |

> **Critical Data Characteristic**: In the raw dataset arrays, all Class 0 images occur contiguously, followed by all Class 1 images. Stratified shuffling with a fixed random seed (`42`) is strictly enforced to prevent severe single-class batch divergence during validation.

---

## 4. Data Preprocessing Pipeline

### A. Dataset Array Inversion
The `.npy` arrays from PEER Hub ImageNet were stored with **Caffe-style BGR ImageNet mean subtraction**:
$$\mu_{BGR} = [103.939, 116.779, 123.68]$$
* **Observed Value Range in Arrays**: $[-123.68, 151.06]$
* **Restoration Equation**:
  $$\text{RGB}_{[0, 255]} = \text{clip}\left((\text{Image}_{BGR} + \mu_{BGR}), 0, 255\right)_{[\dots, ::-1]}$$
* **Model Input Mapping**:
  $$\text{Input}_{[-1, 1]} = \frac{\text{RGB}_{[0, 255]}}{127.5} - 1.0$$

### B. Uploaded Inference Preprocessing (Streamlit App)
Uploaded user photographs (JPG/PNG) are standard RGB images in $[0, 255]$. They do **not** require BGR mean addition. The web application performs Lanczos resizing to $(224, 224)$, ensures 3-channel RGB format, and scales directly to $[-1.0, 1.0]$ via $(x / 127.5) - 1.0$.

---

## 5. Methodology & Architecture

### Transfer Learning Strategy
Due to the computational limits of consumer dual-core laptops (Intel Core i3-5005U with 8 GB RAM), training a deep convolutional network from scratch or backpropagating end-to-end causes system lockups and thermal throttling.

We adopt **transfer learning with bottleneck feature caching**:
1. **Backbone**: MobileNetV2 pretrained on ImageNet instantiated without top classification layers (`include_top=False`).
2. **Frozen Weights**: All convolutional layers are frozen (`trainable = False`), preserving 2,257,984 pretrained weights without backpropagation overhead.
3. **Global Average Pooling**: Compresses spatial feature maps into a 1,280-dimensional feature vector per image.
4. **Resumable Disk Caching**: Features are written directly to disk via `np.lib.format.open_memmap` in mini-batches of 8, with an atomic `.meta.json` progress sidecar. Working RAM remains below 1.5 GB.
5. **Trainable Classification Head**:
   * Input: 1,280-dim feature vector
   * Dropout layer ($p = 0.4$)
   * Dense layer (64 units, ReLU activation, 81,984 parameters)
   * Output Dense layer (2 units, Softmax activation, 130 parameters)
   * Trainable parameters: **82,114**
   * Total model parameters: **2,340,098** (8.93 MB)
6. **Optimization**:
   * Optimizer: Adam ($\text{lr} = 0.001$)
   * Loss: Categorical Cross-Entropy
   * Callbacks: EarlyStopping ($\text{patience}=5$), ReduceLROnPlateau ($\text{factor}=0.5, \text{patience}=2$)

---

## 6. Verified Experimental Results

The model converged in 19 epochs (EarlyStopping restored weights at epoch 14) and was evaluated on the **official held-out test set of 1,460 images**.

### Summary Performance Metrics

| Evaluation Metric | Measured Value |
| :--- | :---: |
| **Official Test Accuracy** | **84.25%** (1,230 / 1,460 correct) |
| **ROC-AUC (Damaged Target)** | **0.9289** (Ranking & discrimination capacity) |
| **Damaged Precision** | 0.8366 (83.7%) |
| **Damaged Recall** | 0.8591 (85.9%) |
| **Damaged F1-Score** | 0.8477 (84.8%) |
| **Undamaged Precision** | 0.8489 (84.9%) |
| **Undamaged Recall** | 0.8252 (82.5%) |
| **Undamaged F1-Score** | 0.8369 (83.7%) |
| **Macro Average F1-Score** | 0.8423 (84.2%) |
| **Weighted Average F1-Score** | 0.8424 (84.2%) |

### Official Test Set Confusion Matrix

$$\begin{bmatrix} \text{Actual Damaged (745)} \\ \text{Actual Undamaged (715)} \end{bmatrix} \times \begin{bmatrix} \text{Pred Damaged} & \text{Pred Undamaged} \end{bmatrix} = \begin{bmatrix} \mathbf{640} & \mathbf{105} \\ \mathbf{125} & \mathbf{590} \end{bmatrix}$$

* **True Positives ($TP = 640$)**: 85.91% of actual damaged structures correctly flagged.
* **True Negatives ($TN = 590$)**: 82.52% of sound structures correctly cleared.
* **False Positives ($FP = 125$)**: 17.48% false alarm rate.
* **False Negatives ($FN = 105$)**: 14.09% miss rate.
* **Diagnostic Interpretation**: In civil safety triage, Damaged recall (85.91%) is prioritized because missed damage (False Negatives) represents a life-safety hazard. Plausible visual explanations for false negatives include low-contrast hairline fissures and heavy directional shadows; plausible explanations for false alarms include high-contrast brick mortar seams and external conduits.

---

## 7. Project Directory Structure

```text
ml-miniproject/
├── README.md                           # Master documentation
├── requirements.txt                    # Validated dependencies
├── .gitignore                          # Excludes datasets, caches, and local artifacts
├── app.py                              # Streamlit web application
├── configs/
│   └── config.yaml                     # Pipeline configuration
├── docs/
│   ├── report.pdf                      # Submission-ready technical report (PDF)
│   └── Structural_Damage_Recognition_Presentation.pptx # Review presentation slide deck
├── models/
│   └── mobilenet_v2_damage_classifier.keras # Trained unified Keras model (~10.1 MB)
├── outputs/
│   ├── metrics/                        # test_metrics.json, classification_report.txt, training_history.json
│   └── plots/                          # confusion_matrix.png, roc_curve.png, training_curves.png
├── scripts/
│   ├── inspect_dataset.py              # Dataset sanity check & sample visualization
│   ├── train_local_cached.py           # Resumable feature extraction & classifier training
├── src/structural_damage/
│   ├── __init__.py                     # Module interface
│   ├── preprocessing.py                # Caffe restoration & inference normalization
│   ├── dataset.py                      # Memory-mapped loader & stratified splitter
│   ├── batch_generator.py              # Sequence batch generator
│   ├── models.py                       # Keras model definitions
│   ├── evaluate.py                     # Metric calculation & plotting utilities
│   ├── predict.py                      # Single-image prediction helper
│   └── train_local_cached.py           # Core training pipeline implementation
├── tests/
│   ├── test_preprocessing.py           # Unit tests for image inversion & normalization
│   ├── test_labels.py                  # Unit tests for one-hot mapping & stratification
│   ├── test_batch_loader.py            # Unit tests for batch generator
│   ├── test_dataset_metadata.py        # Unit tests for header validation
│   └── test_resumable_cache.py         # Unit tests for resumable extraction
```

---

## 8. Setup & Execution Instructions

### A. Environment Setup
```bash
# Open Anaconda Prompt / Terminal
conda create -n damage-ml python=3.10 -y
conda activate damage-ml

# Install validated dependencies
pip install -r requirements.txt
```

### B. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The app loads `models/mobilenet_v2_damage_classifier.keras` into memory via `@st.cache_resource` and performs single-photograph inference.

### C. Run the Unit Test Suite
```bash
python -m pytest tests/ -v
```
All 29 tests execute across preprocessing, dataset splitting, batch loading, and cache handling.

### D. Dataset Inspection & Training Reproduction
```bash
# 1. Inspect dataset arrays (auto-discovers temp_local/data or data/)
python scripts/inspect_dataset.py

# 2. Reproduce feature extraction & training (resumes from disk if cache exists)
python scripts/train_local_cached.py
```

---

## 9. Limitations & Ethical Considerations

1. **Binary Classification Scope**: The model provides coarse `Damaged` vs. `Undamaged` screening; it does not classify structural damage type (e.g., flexural, shear, buckling) or quantify severity grades.
2. **False Negatives in Safety-Critical Scenarios**: 105 damaged structures were misclassified as undamaged (14.09% miss rate). In civil engineering, missed damage is high-risk.
3. **Resolution Bounds**: Images are resized to $224 \times 224$. Micro-hairline cracks may be attenuated by downsampling.
4. **Safety Disclaimer**: This tool is an academic research prototype for educational evaluation and must **never** be used as a substitute for certified civil/structural engineering inspections.

---

## 10. Team Member Contributions

| Student Name | SRN | Primary Contributions |
| :--- | :--- | :--- |
| **MIRUNJAI SURESH KUMAR** | `PES1UG24AM161` | Dataset auditing, Caffe BGR preprocessing restoration, memory-mapped batch loading, MobileNetV2 feature-caching pipeline implementation. |
| **MOHAMMED SINAN M T** | `PES1UG24AM165` | Classification head optimization, evaluation metrics calculation, Streamlit web app design, technical report & presentation authoring. |
