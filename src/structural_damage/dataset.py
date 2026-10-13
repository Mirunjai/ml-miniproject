"""Memory-efficient dataset loading and batch generation.

Designed for the PHI-Net Task 2 dataset where the training array
is ~6.8 GiB. Uses numpy memory mapping to avoid loading the entire
array into RAM.
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple
from sklearn.model_selection import train_test_split

def load_dataset(data_dir: Union[str, Path]) -> Dict:
    """Load dataset with memory mapping for images.
    
    Args:
        data_dir: Directory containing the .npy files.
        
    Returns:
        Dict with keys: 'X_train', 'y_train', 'X_test', 'y_test', and 'metadata'.
    """
    data_dir = Path(data_dir)
    
    files = {
        'X_train': data_dir / 'task2_X_train.npy',
        'y_train': data_dir / 'task2_y_train.npy',
        'X_test': data_dir / 'task2_X_test.npy',
        'y_test': data_dir / 'task2_y_test.npy',
    }
    
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required dataset file: {path}")
            
    X_train = np.load(files['X_train'], mmap_mode='r')
    X_test = np.load(files['X_test'], mmap_mode='r')
    
    y_train = np.load(files['y_train'])
    y_test = np.load(files['y_test'])
    
    if len(X_train) != len(y_train):
        raise ValueError("Train images count != train labels count")
    if len(X_test) != len(y_test):
        raise ValueError("Test images count != test labels count")
        
    if X_train.shape[1:] != (224, 224, 3) or X_test.shape[1:] != (224, 224, 3):
        raise ValueError("Image shape must be (N, 224, 224, 3)")
        
    if y_train.shape[1] != 2 or y_test.shape[1] != 2:
        raise ValueError("Labels must be one-hot (N, 2)")
        
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'metadata': {
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    }

def create_train_val_split(y_train: np.ndarray, val_fraction: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create a stratified train/validation split.
    
    Returns:
        (train_indices, val_indices) as sorted arrays.
    """
    class_ids = np.argmax(y_train, axis=1)
    indices = np.arange(len(y_train))
    
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, random_state=seed, stratify=class_ids
    )
    
    train_idx.sort()
    val_idx.sort()
    
    train_classes = class_ids[train_idx]
    val_classes = class_ids[val_idx]
    
    assert len(np.unique(train_classes)) == 2
    assert len(np.unique(val_classes)) == 2
    
    print(f"Train split: {len(train_idx)} samples")
    print(f"Val split: {len(val_idx)} samples")
    
    return train_idx, val_idx


try:
    from tensorflow.keras.utils import Sequence
except ImportError:
    class Sequence:
        pass


class BatchGenerator(Sequence):
    """Memory-efficient batch generator."""
    
    def __init__(self, X_mmap, y_labels, indices=None, batch_size=32, preprocessing_fn=None, shuffle=True, seed=42):
        self.X_mmap = X_mmap
        self.y_labels = y_labels
        self.indices = np.arange(len(X_mmap)) if indices is None else np.array(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        
        if preprocessing_fn is None:
            from structural_damage.preprocessing import dataset_to_inception
            self.preprocessing_fn = dataset_to_inception
        else:
            self.preprocessing_fn = preprocessing_fn
            
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
        
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.indices))
        
        batch_indices = self.indices[start_idx:end_idx]
        
        X_batch = self.X_mmap[batch_indices]
        y_batch = self.y_labels[batch_indices]
        
        if self.preprocessing_fn:
            X_batch = self.preprocessing_fn(X_batch)
            
        return X_batch, y_batch
        
    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
