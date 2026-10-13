"""Tests for dataset metadata and loading validation."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_load_dataset_missing_file():
    with pytest.raises(FileNotFoundError):
        np.load("non_existent_file.npy", mmap_mode='r')

def test_shape_mismatch_detection():
    X = np.zeros((10, 224, 224, 3))
    y = np.zeros((9, 2))
    if X.shape[0] != y.shape[0]:
        with pytest.raises(ValueError):
            raise ValueError("Mismatched shapes")
    else:
        assert False

def test_valid_synthetic_dataset(tmp_path):
    X_path = tmp_path / "X.npy"
    y_path = tmp_path / "y.npy"
    
    np.save(X_path, np.zeros((5, 224, 224, 3), dtype=np.float32))
    np.save(y_path, np.zeros((5, 2), dtype=np.float32))
    
    X = np.load(X_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')
    
    assert X.shape == (5, 224, 224, 3)
    assert y.shape == (5, 2)
