"""Tests for batch generation."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

pytest.importorskip('tensorflow')

try:
    from structural_damage.batch_generator import BatchGenerator
except ImportError:
    pass

@pytest.fixture
def synthetic_data():
    X = np.random.uniform(0, 255, (20, 224, 224, 3)).astype(np.float32)
    y = np.array([[1, 0]] * 10 + [[0, 1]] * 10).astype(np.float32)
    return X, y

def test_batch_shapes(synthetic_data):
    try:
        from structural_damage.batch_generator import BatchGenerator
    except ImportError:
        pytest.skip("BatchGenerator not implemented")
        
    X, y = synthetic_data
    gen = BatchGenerator(X, y, batch_size=8, shuffle=False)
    X_batch, y_batch = gen[0]
    assert X_batch.shape == (8, 224, 224, 3)
    assert y_batch.shape == (8, 2)

def test_all_samples_covered(synthetic_data):
    try:
        from structural_damage.batch_generator import BatchGenerator
    except ImportError:
        pytest.skip("BatchGenerator not implemented")
        
    X, y = synthetic_data
    gen = BatchGenerator(X, y, batch_size=8, shuffle=False)
    assert len(gen) == 3 # 20 / 8 = 2.5 -> 3 batches
    total_samples = sum(len(gen[i][0]) for i in range(len(gen)))
    assert total_samples == 20

def test_last_batch_size(synthetic_data):
    try:
        from structural_damage.batch_generator import BatchGenerator
    except ImportError:
        pytest.skip("BatchGenerator not implemented")
        
    X, y = synthetic_data
    gen = BatchGenerator(X, y, batch_size=8, shuffle=False)
    X_batch, _ = gen[2]
    assert X_batch.shape[0] == 4

def test_labels_match_images(synthetic_data):
    try:
        from structural_damage.batch_generator import BatchGenerator
    except ImportError:
        pytest.skip("BatchGenerator not implemented")
        
    X, y = synthetic_data
    gen = BatchGenerator(X, y, batch_size=20, shuffle=False)
    X_batch, y_batch = gen[0]
    assert np.array_equal(y_batch, y)

def test_shuffle_changes_order(synthetic_data):
    try:
        from structural_damage.batch_generator import BatchGenerator
    except ImportError:
        pytest.skip("BatchGenerator not implemented")
        
    X, y = synthetic_data
    indices = np.arange(20)
    gen = BatchGenerator(X, y, batch_size=20, shuffle=True)
    gen.indices = indices.copy()
    gen.on_epoch_end()
    assert not np.array_equal(gen.indices, indices)

def test_no_shuffle_preserves_order(synthetic_data):
    try:
        from structural_damage.batch_generator import BatchGenerator
    except ImportError:
        pytest.skip("BatchGenerator not implemented")
        
    X, y = synthetic_data
    indices = np.arange(20)
    gen = BatchGenerator(X, y, batch_size=20, shuffle=False)
    gen.indices = indices.copy()
    gen.on_epoch_end()
    assert np.array_equal(gen.indices, indices)
