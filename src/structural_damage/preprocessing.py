"""Preprocessing utilities for PHI-Net structural damage images.

The dataset arrays store images as float32 with Caffe-style BGR
mean subtraction applied:
  - Channel order: BGR (not RGB)
  - Each channel has had its ImageNet mean subtracted:
    B: 103.939, G: 116.779, R: 123.68
  - Pixel range after subtraction: approximately [-124, 151]

This module provides functions to:
  1. Reverse the preprocessing for visualization
  2. Convert dataset arrays to model-ready inputs
  3. Prepare ordinary RGB images (uploads) for inference
"""

import numpy as np
from typing import Union, Tuple

CAFFE_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)
LABEL_NAMES = {0: 'Damaged', 1: 'Undamaged'}
NUM_CLASSES = 2

def restore_for_visualization(images: np.ndarray) -> np.ndarray:
    """Reverse Caffe preprocessing for visualization.
    
    Args:
        images: Preprocessed BGR images from dataset (batch or single).
        
    Returns:
        RGB uint8 images suitable for display.
    """
    img_copy = images.copy()
    img_copy += CAFFE_MEAN_BGR
    img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
    
    # Reverse BGR to RGB
    if img_copy.ndim == 3:
        return img_copy[..., ::-1]
    elif img_copy.ndim == 4:
        return img_copy[..., ::-1]
    else:
        raise ValueError("Images must be 3D or 4D array")

def dataset_to_inception(images: np.ndarray) -> np.ndarray:
    """Convert dataset Caffe-preprocessed BGR images to InceptionV3 format.
    
    This is equivalent to tf.keras.applications.inception_v3.preprocess_input.
    
    Args:
        images: Preprocessed BGR images from dataset.
        
    Returns:
        float32 array with Inception preprocessing.
    """
    img_copy = images.copy()
    img_copy += CAFFE_MEAN_BGR
    
    # Reverse BGR to RGB
    if img_copy.ndim == 3:
        rgb = img_copy[..., ::-1]
    elif img_copy.ndim == 4:
        rgb = img_copy[..., ::-1]
    else:
        raise ValueError("Images must be 3D or 4D array")
        
    # InceptionV3 preprocessing
    return (rgb / 127.5) - 1.0

def prepare_upload_for_inception(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Prepare ordinary RGB image (from upload) for Inception inference.
    
    Args:
        image: Ordinary RGB uint8 image.
        target_size: Target size.
        
    Returns:
        float32 array shape (1, H, W, 3).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a 3-channel RGB image")
        
    try:
        from PIL import Image
        img_pil = Image.fromarray(image)
        if img_pil.size != target_size:
            img_pil = img_pil.resize(target_size, Image.BILINEAR)
        resized = np.array(img_pil)
    except ImportError:
        import cv2
        if (image.shape[1], image.shape[0]) != target_size:
            resized = cv2.resize(image, target_size)
        else:
            resized = image
            
    resized = resized.astype(np.float32)
    processed = (resized / 127.5) - 1.0
    return np.expand_dims(processed, axis=0)

def convert_onehot_to_labels(y_onehot: np.ndarray) -> np.ndarray:
    """Convert one-hot encoded labels to class IDs.
    
    Args:
        y_onehot: Array of shape (N, 2).
        
    Returns:
        Array of shape (N,) with integer class IDs.
    """
    if y_onehot.ndim != 2 or y_onehot.shape[1] != 2:
        raise ValueError("Input must have shape (N, 2)")
        
    if not np.all(np.isclose(np.sum(y_onehot, axis=1), 1.0)):
        raise ValueError("Rows must sum to 1")
        
    if not np.all(np.isin(y_onehot, [0, 1])):
        raise ValueError("Values must be 0 or 1")
        
    return np.argmax(y_onehot, axis=1)
