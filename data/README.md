# Dataset Directory

This directory should contain the PHI-Net Task 2 dataset files.

## Required Files

| File | Description | Shape | Size |
|------|-------------|-------|------|
| `task2_X_train.npy` | Training images | (11811, 224, 224, 3) float32 | ~6.8 GiB |
| `task2_y_train.npy` | Training labels (one-hot) | (11811, 2) float32 | ~92 KiB |
| `task2_X_test.npy` | Test images | (1460, 224, 224, 3) float32 | ~838 MiB |
| `task2_y_test.npy` | Test labels (one-hot) | (1460, 2) float32 | ~12 KiB |

## Label Encoding

- Column 0: Damaged (D) — `[1, 0]`
- Column 1: Undamaged (UD) — `[0, 1]`

## Source

PEER Hub ImageNet (PHI-Net) — Task 2: Damage State  
Author: Yuqing Gao (gaoyuqing@berkeley.edu)  
License: CC BY-NC-SA 4.0

## Note

These files are NOT committed to git due to their size.
See the project README for instructions on obtaining the dataset.
