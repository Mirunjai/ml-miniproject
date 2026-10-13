"""Inference utilities for structural damage prediction."""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union, Dict

from structural_damage.preprocessing import prepare_upload_for_inception, LABEL_NAMES

def load_trained_model(model_path: Union[str, Path]) -> tf.keras.Model:
    """Load a trained Keras model."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    return tf.keras.models.load_model(model_path)

def predict_image(model: tf.keras.Model, image: np.ndarray, target_size=(224, 224)) -> Dict:
    """Run prediction on a single image."""
    processed_image = prepare_upload_for_inception(image, target_size=target_size)
    
    preds = model.predict(processed_image, verbose=0)[0]
    
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])
    class_name = LABEL_NAMES[class_id]
    
    return {
        'class_name': class_name,
        'class_id': class_id,
        'confidence': confidence,
        'probabilities': {
            LABEL_NAMES[0]: float(preds[0]),
            LABEL_NAMES[1]: float(preds[1])
        }
    }
