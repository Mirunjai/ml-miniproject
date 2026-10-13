"""Tests for label handling and splitting."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from structural_damage.preprocessing import convert_onehot_to_labels
from sklearn.model_selection import train_test_split

def test_onehot_to_class_id_mapping():
    labels = np.array([[1, 0], [0, 1]])
    res = convert_onehot_to_labels(labels)
    assert np.array_equal(res, [0, 1])

def test_class_balance_preserved_in_split():
    labels = np.array([[1, 0]] * 10 + [[0, 1]] * 90) # Imbalanced
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    train_labels = convert_onehot_to_labels(labels[train_idx])
    val_labels = convert_onehot_to_labels(labels[val_idx])
    
    assert 0 in train_labels and 1 in train_labels
    assert 0 in val_labels and 1 in val_labels

def test_split_reproducibility():
    labels = np.array([[1, 0]] * 50 + [[0, 1]] * 50)
    indices = np.arange(len(labels))
    train1, _ = train_test_split(indices, test_size=0.2, random_state=42)
    train2, _ = train_test_split(indices, test_size=0.2, random_state=42)
    assert np.array_equal(train1, train2)

def test_split_no_overlap():
    indices = np.arange(100)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    assert len(set(train_idx).intersection(set(val_idx))) == 0

def test_split_covers_all():
    indices = np.arange(100)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    assert set(train_idx).union(set(val_idx)) == set(indices)
