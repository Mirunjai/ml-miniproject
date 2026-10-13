"""Tests for preprocessing utilities."""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from structural_damage.preprocessing import (
    CAFFE_MEAN_BGR,
    restore_for_visualization,
    dataset_to_inception,
    prepare_upload_for_inception,
    convert_onehot_to_labels,
    LABEL_NAMES,
    NUM_CLASSES,
)

def test_caffe_mean_values():
    assert np.allclose(CAFFE_MEAN_BGR, [103.939, 116.779, 123.68], atol=1e-4)

def test_restore_single_image():
    img = np.zeros((224, 224, 3), dtype=np.float32)
    restored = restore_for_visualization(img)
    assert restored.dtype == np.uint8
    assert restored.shape == (224, 224, 3)
    assert np.allclose(restored[0, 0], [124, 117, 104], atol=1)

def test_restore_batch():
    batch = np.zeros((4, 224, 224, 3), dtype=np.float32)
    restored = restore_for_visualization(batch)
    assert restored.dtype == np.uint8
    assert restored.shape == (4, 224, 224, 3)
    assert np.allclose(restored[0, 0, 0], [124, 117, 104], atol=1)

def test_dataset_to_inception_range():
    batch = np.random.uniform(0, 255, size=(4, 224, 224, 3)).astype(np.float32)
    batch[:, :, :, 0] -= CAFFE_MEAN_BGR[0]
    batch[:, :, :, 1] -= CAFFE_MEAN_BGR[1]
    batch[:, :, :, 2] -= CAFFE_MEAN_BGR[2]
    out = dataset_to_inception(batch)
    assert np.min(out) >= -1.0
    assert np.max(out) <= 1.0

def test_dataset_to_inception_zero_input():
    batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
    out = dataset_to_inception(batch)
    assert np.min(out) >= -1.0 and np.max(out) <= 1.0

def test_prepare_upload_rgb_image():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    out = prepare_upload_for_inception(img)
    assert out.shape == (1, 224, 224, 3)
    assert np.min(out) >= -1.0 and np.max(out) <= 1.0

def test_prepare_upload_already_correct_size():
    img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    out = prepare_upload_for_inception(img)
    assert out.shape == (1, 224, 224, 3)
    assert np.min(out) >= -1.0 and np.max(out) <= 1.0

def test_convert_onehot_valid():
    labels = np.array([[1, 0], [0, 1], [1, 0]])
    res = convert_onehot_to_labels(labels)
    assert np.array_equal(res, [0, 1, 0])

def test_convert_onehot_invalid_shape():
    labels = np.array([[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError):
        convert_onehot_to_labels(labels)

def test_convert_onehot_invalid_values():
    labels = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError):
        convert_onehot_to_labels(labels)

def test_label_names():
    assert LABEL_NAMES == {0: 'Damaged', 1: 'Undamaged'}
    assert NUM_CLASSES == 2

def test_roundtrip_consistency():
    # Original RGB image in [0, 255]
    orig_rgb = np.random.randint(10, 245, (224, 224, 3), dtype=np.uint8)
    # Convert to BGR float32 and subtract ImageNet BGR means
    bgr_float = orig_rgb[..., ::-1].astype(np.float32) - CAFFE_MEAN_BGR
    # Invert via restore_for_visualization
    restored_rgb = restore_for_visualization(bgr_float)
    # Differences must be at most 1 due to rounding/clipping
    assert np.all(np.abs(orig_rgb.astype(int) - restored_rgb.astype(int)) <= 1)
