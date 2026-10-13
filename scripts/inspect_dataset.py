import argparse
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_file_size_mib(filepath: Path) -> float:
    return filepath.stat().st_size / (1024 * 1024)

def restore_for_visualization(img_bgr: np.ndarray) -> np.ndarray:
    """Restores Caffe-style BGR mean-subtracted image to RGB uint8 for visualization."""
    bgr_means = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    # Add means
    img = img_bgr + bgr_means
    # Clip to [0, 255]
    img = np.clip(img, 0, 255).astype(np.uint8)
    # Reverse BGR to RGB
    img = img[..., ::-1]
    return img

def inspect_dataset(data_dir: Path, output_dir: Path) -> bool:
    print(f"Inspecting dataset in {data_dir}...")
    
    files = {
        "X_train": data_dir / "task2_X_train.npy",
        "y_train": data_dir / "task2_y_train.npy",
        "X_test": data_dir / "task2_X_test.npy",
        "y_test": data_dir / "task2_y_test.npy"
    }
    
    missing_files = [f for name, f in files.items() if not f.exists()]
    if missing_files:
        print(f"Error: Missing files: {missing_files}")
        return False
        
    for name, f in files.items():
        size = get_file_size_mib(f)
        arr = np.load(f, mmap_mode='r')
        print(f"{name} ({f.name}): size={size:.2f} MiB, shape={arr.shape}, dtype={arr.dtype}")
    
    # 5. Load labels fully
    y_train = np.load(files["y_train"])
    y_test = np.load(files["y_test"])
    
    for split_name, y in [("train", y_train), ("test", y_test)]:
        print(f"Validating {split_name} labels...")
        if len(y.shape) != 2 or y.shape[1] != 2:
            print(f"Error: {split_name} labels shape is not (N, 2)")
            return False
            
        row_sums = y.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            print(f"Error: {split_name} labels are not valid one-hot (rows don't sum to 1)")
            return False
            
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            print(f"Error: {split_name} labels contain values other than 0 and 1")
            return False
            
        damaged_count = int(y[:, 0].sum())
        undamaged_count = int(y[:, 1].sum())
        print(f"  Class 0 (Damaged): {damaged_count}")
        print(f"  Class 1 (Undamaged): {undamaged_count}")

    # 6. Check image arrays (first 500 images)
    for split_name, x_file in [("train", files["X_train"]), ("test", files["X_test"])]:
        print(f"Validating {split_name} images...")
        x = np.load(x_file, mmap_mode='r')
        if x.shape[1:] != (224, 224, 3):
            print(f"Error: {split_name} image dimensions are not (224, 224, 3)")
            return False
            
        # Check image_count == label_count
        y = y_train if split_name == "train" else y_test
        if x.shape[0] != y.shape[0]:
            print(f"Error: {split_name} image count ({x.shape[0]}) != label count ({y.shape[0]})")
            return False
            
        num_to_check = min(500, x.shape[0])
        batch_size = 100
        min_val, max_val = float('inf'), float('-inf')
        channel_sums = np.zeros(3)
        pixel_count = 0
        
        has_nan = False
        has_inf = False
        
        for i in range(0, num_to_check, batch_size):
            batch = x[i:i+batch_size]
            
            if np.isnan(batch).any(): has_nan = True
            if np.isinf(batch).any(): has_inf = True
            
            min_val = min(min_val, batch.min())
            max_val = max(max_val, batch.max())
            
            channel_sums += batch.sum(axis=(0, 1, 2))
            pixel_count += batch.shape[0] * batch.shape[1] * batch.shape[2]
            
        if has_nan or has_inf:
            print(f"Error: {split_name} images contain NaN or Inf values")
            return False
            
        channel_means = channel_sums / pixel_count
        print(f"  Range: [{min_val:.3f}, {max_val:.3f}]")
        print(f"  Mean per channel: {channel_means}")

    # 9. Sample images
    print("Sampling images for visualization...")
    X_test = np.load(files["X_test"], mmap_mode='r')
    
    damaged_indices = np.where(y_test[:, 0] == 1)[0]
    undamaged_indices = np.where(y_test[:, 1] == 1)[0]
    
    # Randomly select 6 of each
    np.random.seed(42)
    selected_damaged = np.random.choice(damaged_indices, 6, replace=False)
    selected_undamaged = np.random.choice(undamaged_indices, 6, replace=False)
    
    # 11. Plot and save
    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    axes = axes.flatten()
    
    selected_indices = list(selected_damaged) + list(selected_undamaged)
    
    for i, idx in enumerate(selected_indices):
        img = X_test[idx]
        img_vis = restore_for_visualization(img)
        
        ax = axes[i]
        ax.imshow(img_vis)
        label = "Damaged" if i < 6 else "Undamaged"
        ax.set_title(f"Test Idx {idx}\n{label}")
        ax.axis('off')
        
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sample_images.png"
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")
    
    print("\nSummary: All checks passed!")
    return True

if __name__ == "__main__":
    default_data = "temp_local/data" if Path("temp_local/data/task2_X_train.npy").exists() else ("data" if Path("data/task2_X_train.npy").exists() else ".")
    parser = argparse.ArgumentParser(description="Inspect PHI-Net Task 2 dataset.")
    parser.add_argument("--data-dir", type=str, default=default_data, help="Directory containing .npy files")
    parser.add_argument("--output-dir", type=str, default="outputs/plots", help="Directory for output visualizations")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    success = inspect_dataset(data_dir, output_dir)
    if not success:
        sys.exit(1)
