"""Evaluation metrics, plots, and error analysis."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, roc_curve
)
from typing import Dict, List, Optional, Callable
import json

def compute_metrics(y_true_onehot: np.ndarray, y_pred_probs: np.ndarray, class_names: Optional[List[str]] = None) -> Dict:
    """Compute comprehensive evaluation metrics."""
    if class_names is None:
        class_names = ['Damaged', 'Undamaged']
        
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    roc_auc = roc_auc_score((y_true == 0).astype(int), y_pred_probs[:, 0])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'roc_auc_damaged': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'per_class': {}
    }
    
    for i, name in enumerate(class_names):
        metrics['per_class'][name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i])
        }
        
    return metrics

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Optional[str] = None):
    """Plot confusion matrix heatmap."""
    cm_array = np.array(cm)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm_array.max() / 2.
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            plt.text(j, i, format(cm_array[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_array[i, j] > thresh else "black")
                     
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_training_history(history_dict: Dict, save_path: Optional[str] = None):
    """Plot loss and accuracy training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    if 'loss' in history_dict:
        ax1.plot(history_dict['loss'], label='Train')
    if 'val_loss' in history_dict:
        ax1.plot(history_dict['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    if 'accuracy' in history_dict:
        ax2.plot(history_dict['accuracy'], label='Train')
    if 'val_accuracy' in history_dict:
        ax2.plot(history_dict['val_accuracy'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true_onehot: np.ndarray, y_pred_probs: np.ndarray, save_path: Optional[str] = None):
    """Plot ROC curve for Damaged class."""
    y_true_damaged = (np.argmax(y_true_onehot, axis=1) == 0).astype(int)
    y_prob_damaged = y_pred_probs[:, 0]
    
    fpr, tpr, _ = roc_curve(y_true_damaged, y_prob_damaged)
    roc_auc = roc_auc_score(y_true_damaged, y_prob_damaged)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Damaged Class)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def error_analysis(X_data: np.ndarray, y_true_onehot: np.ndarray, y_pred_probs: np.ndarray, 
                  restore_fn: Callable, num_examples: int = 5, save_path: Optional[str] = None):
    """Perform error analysis and plot examples."""
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    fp_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    fn_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    
    fp_idx = fp_idx[:num_examples]
    fn_idx = fn_idx[:num_examples]
    
    # Example logic omitted for brevity
    print(f"False Positives found: {len(fp_idx)}, False Negatives: {len(fn_idx)}")
    
def save_metrics_json(metrics_dict: Dict, save_path: str):
    """Save metrics dict to JSON."""
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
