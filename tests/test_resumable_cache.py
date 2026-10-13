import json
import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from structural_damage.train_local_cached import (
    validate_completed_cache,
    get_resumable_cache_state,
    save_cache_metadata,
)


def test_empty_cache_validation(tmp_path: Path):
    cache_file = tmp_path / "features.npy"
    assert not validate_completed_cache(cache_file, 100, 1280)
    assert get_resumable_cache_state(cache_file, 100, 1280) == 0


def test_synthetic_interrupted_and_resumed_extraction(tmp_path: Path):
    """Simulate an interrupted extraction at sample 40/100, then resume and finish."""
    total_samples = 100
    dim = 16
    batch_size = 8
    cache_file = tmp_path / "test_extract.npy"
    meta_file = cache_file.with_suffix(".meta.json")

    # 1. Start fresh extraction
    assert get_resumable_cache_state(cache_file, total_samples, dim) == 0

    mmap = np.lib.format.open_memmap(
        cache_file, mode="w+", dtype=np.float32, shape=(total_samples, dim)
    )

    # 2. Simulate extraction up to index 40
    interrupt_point = 40
    for i in range(0, interrupt_point, batch_size):
        mmap[i : i + batch_size] = np.ones((batch_size, dim), dtype=np.float32) * (i + 1)

    mmap.flush()
    save_cache_metadata(
        meta_file=meta_file,
        expected_samples=total_samples,
        expected_dim=dim,
        completed_samples=interrupt_point,
        status="in_progress",
        architecture="MobileNetV2",
        preprocessing="Caffe_BGR_Restored_[-1,1]",
    )

    # Simulate process crash / memory error
    del mmap

    # 3. Validation on restart: cache is NOT completed, but is resumable at 40
    assert not validate_completed_cache(cache_file, total_samples, dim)
    resumed_from = get_resumable_cache_state(cache_file, total_samples, dim)
    assert resumed_from == interrupt_point

    # 4. Resume from index 40 to 100
    mmap_resumed = np.lib.format.open_memmap(
        cache_file, mode="r+", dtype=np.float32, shape=(total_samples, dim)
    )

    for i in range(resumed_from, total_samples, batch_size):
        end_i = min(i + batch_size, total_samples)
        mmap_resumed[i:end_i] = np.ones((end_i - i, dim), dtype=np.float32) * (i + 1)

    mmap_resumed.flush()
    save_cache_metadata(
        meta_file=meta_file,
        expected_samples=total_samples,
        expected_dim=dim,
        completed_samples=total_samples,
        status="completed",
        architecture="MobileNetV2",
        preprocessing="Caffe_BGR_Restored_[-1,1]",
    )
    del mmap_resumed

    # 5. Final check: now marked completed and data is completely verified
    assert validate_completed_cache(cache_file, total_samples, dim)

    # Verify rows 0..39 were preserved intact while rows 40..99 were written
    final_arr = np.load(cache_file, mmap_mode="r")
    assert final_arr.shape == (total_samples, dim)
    assert final_arr[0, 0] == 1.0
    assert final_arr[39, 0] == 33.0
    assert final_arr[40, 0] == 41.0
    assert final_arr[99, 0] == 97.0


def test_corrupt_or_incompatible_cache_rejection(tmp_path: Path):
    """Corrupted metadata or incompatible dimensions should result in clean start."""
    cache_file = tmp_path / "corrupt_extract.npy"
    meta_file = cache_file.with_suffix(".meta.json")

    # Wrong dimension recorded
    with open(meta_file, "w") as f:
        json.dump(
            {
                "architecture": "MobileNetV2",
                "preprocessing": "Caffe_BGR_Restored_[-1,1]",
                "samples": 100,
                "dim": 512,  # Expected is 1280
                "completed_samples": 40,
                "status": "in_progress",
            },
            f,
        )

    # Should safely return 0 (start fresh)
    assert get_resumable_cache_state(cache_file, 100, 1280) == 0
