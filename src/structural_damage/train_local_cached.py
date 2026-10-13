"""
Local CPU Feature-Cached Training Pipeline for MobileNetV2.
Specially tuned for Intel Core i3-5005U with 8GB RAM.

Resumable extraction architecture:
  1. Creates disk-backed memmap arrays (np.lib.format.open_memmap) in outputs/cache/
  2. Tracks progress with checkpoint metadata sidecar (.meta.json) recording
     last completed sample index, sample counts, shape, and architecture.
  3. Periodically flushes array to disk and updates progress.
  4. On interruption or restart, verifies sidecar and array integrity, resumes
     from the exact next batch without re-extracting completed batches.
  5. Runs garbage collection and backend memory cleanup periodically to prevent
     Keras/TF CPU tensor accumulation over long extraction loops.
  6. Safely promotes completed extraction to verified cache only after 100% completion.
"""

import os
import sys
import gc
import time
import json
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Optional but recommended for system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import tensorflow as tf
except ImportError:
    tf = None

# ── Paths ──────────────────────────────────────────────────────────────
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent

# Configurable paths with temp_local and root defaults
data_env = os.environ.get("STRUCTURAL_DAMAGE_DATA_DIR")
if data_env:
    DATA_DIR = Path(data_env)
elif (WORKSPACE_DIR / "temp_local" / "data" / "task2_X_train.npy").exists():
    DATA_DIR = WORKSPACE_DIR / "temp_local" / "data"
elif (WORKSPACE_DIR / "data" / "task2_X_train.npy").exists():
    DATA_DIR = WORKSPACE_DIR / "data"
else:
    DATA_DIR = WORKSPACE_DIR

cache_env = os.environ.get("STRUCTURAL_DAMAGE_CACHE_DIR")
if cache_env:
    CACHE_DIR = Path(cache_env)
elif (WORKSPACE_DIR / "temp_local" / "cache").exists():
    CACHE_DIR = WORKSPACE_DIR / "temp_local" / "cache"
else:
    CACHE_DIR = WORKSPACE_DIR / "outputs" / "cache"

MODELS_DIR = WORKSPACE_DIR / "models"
METRICS_DIR = WORKSPACE_DIR / "outputs" / "metrics"
PLOTS_DIR = WORKSPACE_DIR / "outputs" / "plots"

for d in [CACHE_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Preprocessing constants ───────────────────────────────────────────
MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def preprocess_batch(batch_bgr: np.ndarray) -> np.ndarray:
    """Restore Caffe BGR mean-subtracted images to RGB in [-1, 1] for MobileNetV2."""
    batch_rgb = (np.asarray(batch_bgr, dtype=np.float32) + MEAN_BGR)[:, :, :, ::-1]
    batch_rgb = np.clip(batch_rgb, 0.0, 255.0)
    return (batch_rgb / 127.5) - 1.0


def check_system_resources(min_free_ram_gb: float = 1.0, min_free_disk_gb: float = 1.0):
    """Abort safely if system resources are critically low."""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        free_ram_gb = mem.available / (1024 ** 3)
        if free_ram_gb < min_free_ram_gb:
            raise MemoryError(
                f"Available RAM ({free_ram_gb:.2f} GB) is below safe threshold "
                f"({min_free_ram_gb} GB). Close other applications and retry."
            )

    total, used, free = shutil.disk_usage(str(WORKSPACE_DIR))
    free_disk_gb = free / (1024 ** 3)
    if free_disk_gb < min_free_disk_gb:
        raise IOError(
            f"Free disk space ({free_disk_gb:.2f} GB) is below safe threshold "
            f"({min_free_disk_gb} GB)."
        )


def build_mobilenet_extractor():
    """Build frozen MobileNetV2 feature extractor (1,280 pooled features)."""
    input_tensor = tf.keras.Input(shape=(224, 224, 3))
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base.trainable = False
    x = base(input_tensor, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(input_tensor, pooled, name="MobileNetV2_Extractor")


def run_runtime_pilot(extractor_model, X_mmap: np.ndarray) -> float:
    """Run warm-up + steady-state pilot on 32 images to measure throughput."""
    print("\n--- Running 32-Sample Runtime Pilot ---")
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        print(f"Available RAM: {mem.available / (1024**3):.2f} GB")

    # 1. Warm-up (4 images — lets TF compile kernels)
    warmup_batch = preprocess_batch(np.asarray(X_mmap[:4]))
    _ = extractor_model.predict_on_batch(warmup_batch)

    # 2. Steady-state benchmark (32 images, batch_size=4)
    pilot_size = 32
    batch_size = 4
    start_t = time.perf_counter()
    for i in range(4, 4 + pilot_size, batch_size):
        batch = preprocess_batch(np.asarray(X_mmap[i : i + batch_size]))
        _ = extractor_model.predict_on_batch(batch)
    elapsed = time.perf_counter() - start_t

    throughput = pilot_size / elapsed
    est_train_sec = 11811 / throughput
    est_test_sec = 1460 / throughput
    total_est_min = (est_train_sec + est_test_sec) / 60.0

    print(f"Measured Steady-State Throughput: {throughput:.2f} images/sec")
    print(f"Estimated 11,811 Train Extraction: {est_train_sec / 60:.1f} minutes")
    print(f"Estimated 1,460  Test Extraction:  {est_test_sec / 60:.1f} minutes")
    print(f"Total Projected Pipeline Time:     {total_est_min:.1f} minutes")

    return throughput


def validate_completed_cache(cache_file: Path, expected_samples: int, expected_dim: int) -> bool:
    """Return True if completed cache file and metadata are intact and fully extracted."""
    meta_file = cache_file.with_suffix(".meta.json")
    if not cache_file.exists() or not meta_file.exists():
        return False
    try:
        with open(meta_file, "r") as f:
            meta = json.load(f)
        if (
            meta.get("status") == "completed"
            and meta.get("samples") == expected_samples
            and meta.get("dim") == expected_dim
            and meta.get("completed_samples") == expected_samples
        ):
            data = np.load(cache_file, mmap_mode="r")
            if data.shape == (expected_samples, expected_dim):
                return True
        return False
    except Exception:
        return False


def get_resumable_cache_state(
    cache_file: Path,
    expected_samples: int,
    expected_dim: int,
    architecture: str = "MobileNetV2",
    preprocessing: str = "Caffe_BGR_Restored_[-1,1]",
) -> int:
    """Validate partial disk cache and return the number of completed samples.
    
    Returns:
        completed_samples: int (0 if cache should be started fresh)
    """
    meta_file = cache_file.with_suffix(".meta.json")
    if not cache_file.exists() or not meta_file.exists():
        return 0

    try:
        with open(meta_file, "r") as f:
            meta = json.load(f)

        if (
            meta.get("architecture") == architecture
            and meta.get("preprocessing") == preprocessing
            and meta.get("samples") == expected_samples
            and meta.get("dim") == expected_dim
        ):
            completed = int(meta.get("completed_samples", 0))
            if 0 < completed <= expected_samples:
                # Verify array file on disk has correct header and size
                mmap_check = np.lib.format.open_memmap(
                    cache_file,
                    mode="r",
                    dtype=np.float32,
                    shape=(expected_samples, expected_dim),
                )
                del mmap_check
                return completed
    except Exception as e:
        print(f"⚠️  Partial cache validation warning: {e}. Starting fresh.")

    return 0


def save_cache_metadata(
    meta_file: Path,
    expected_samples: int,
    expected_dim: int,
    completed_samples: int,
    status: str,
    architecture: str = "MobileNetV2",
    preprocessing: str = "Caffe_BGR_Restored_[-1,1]",
):
    """Save metadata sidecar safely with atomic replace."""
    tmp_meta = meta_file.with_suffix(".tmp.json")
    meta_data = {
        "architecture": architecture,
        "preprocessing": preprocessing,
        "samples": expected_samples,
        "dim": expected_dim,
        "completed_samples": completed_samples,
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(tmp_meta, "w") as f:
        json.dump(meta_data, f, indent=2)
    os.replace(tmp_meta, meta_file)


def extract_and_cache(
    extractor_model,
    X_mmap: np.ndarray,
    cache_file: Path,
    batch_size: int = 8,
    flush_interval: int = 200,
) -> np.ndarray:
    """Extract features directly into a disk-backed memory-mapped array with resumption.
    
    Memory Safety:
      - Memory allocated directly on disk via np.lib.format.open_memmap
      - Intermediate batch tensors flushed to disk regularly
      - gc.collect() and Keras backend cleanup run periodically
      - Fully resumes from last completed batch if interrupted
    """
    expected_samples = len(X_mmap)
    expected_dim = 1280
    meta_file = cache_file.with_suffix(".meta.json")

    # 1. Check if already 100% completed
    if validate_completed_cache(cache_file, expected_samples, expected_dim):
        print(f"✅ Valid completed feature cache found: {cache_file.name}. Skipping extraction.")
        return np.lib.format.open_memmap(cache_file, mode="r", dtype=np.float32)

    # 2. Check for resumable partial cache
    completed_samples = get_resumable_cache_state(
        cache_file, expected_samples, expected_dim
    )

    if completed_samples > 0:
        print(
            f"🔄 Resuming extraction for {cache_file.name} from sample "
            f"[{completed_samples}/{expected_samples}] "
            f"({(completed_samples / expected_samples) * 100:.1f}% already done)."
        )
        memmap_mode = "r+"
    else:
        print(
            f"📦 Allocating disk-backed memmap for {expected_samples} samples "
            f"({expected_samples * expected_dim * 4 / (1024**2):.1f} MB) -> {cache_file.name}"
        )
        memmap_mode = "w+"

    check_system_resources()

    # Open disk-backed array (writes directly to file, NOT in Python RAM)
    features_mmap = np.lib.format.open_memmap(
        cache_file,
        mode=memmap_mode,
        dtype=np.float32,
        shape=(expected_samples, expected_dim),
    )

    start_idx = completed_samples
    start_t = time.time()
    last_flush_idx = start_idx

    for i in range(start_idx, expected_samples, batch_size):
        end_i = min(i + batch_size, expected_samples)
        batch_prep = preprocess_batch(np.asarray(X_mmap[i:end_i]))
        
        # Write directly into disk-backed array rows
        features_mmap[i:end_i] = extractor_model.predict_on_batch(batch_prep)
        completed_samples = end_i

        # Periodic flush to disk and metadata update
        if (completed_samples - last_flush_idx) >= flush_interval or completed_samples == expected_samples:
            features_mmap.flush()
            save_cache_metadata(
                meta_file=meta_file,
                expected_samples=expected_samples,
                expected_dim=expected_dim,
                completed_samples=completed_samples,
                status="in_progress" if completed_samples < expected_samples else "completed",
            )
            last_flush_idx = completed_samples

        # Diagnostics & Resource checks every 500 samples
        if i > start_idx and (i % 500 == 0 or (i - start_idx) % 500 < batch_size):
            elapsed = time.time() - start_t
            rate = (i - start_idx) / max(elapsed, 0.001)
            remaining_samples = expected_samples - i
            eta_min = (remaining_samples / max(rate, 0.001)) / 60.0

            free_ram_str = ""
            if HAS_PSUTIL:
                mem = psutil.virtual_memory()
                free_ram_str = f" | Free RAM: {mem.available / (1024**3):.2f} GB"

            print(
                f"  [{i}/{expected_samples}] ({rate:.1f} img/s | ETA: {eta_min:.1f} min{free_ram_str})"
            )
            check_system_resources()

            # Prevent Keras/TF CPU tensor accumulation over long loops
            gc.collect()

    # Final flush and verification
    features_mmap.flush()
    assert completed_samples == expected_samples, f"Incomplete: {completed_samples} != {expected_samples}"
    
    save_cache_metadata(
        meta_file=meta_file,
        expected_samples=expected_samples,
        expected_dim=expected_dim,
        completed_samples=expected_samples,
        status="completed",
    )

    # Release writable handle
    del features_mmap
    gc.collect()

    elapsed_total = time.time() - start_t
    print(f"✅ Verified completed extraction saved to {cache_file} in {elapsed_total:.1f}s.")

    # Return read-only memory map for subsequent training
    return np.lib.format.open_memmap(cache_file, mode="r", dtype=np.float32)


def main():
    check_system_resources()

    # ── 1. Memory-map source arrays ───────────────────────────────────
    print(f"Loading memory-mapped arrays from: {DATA_DIR}...")
    X_train_mmap = np.load(
        DATA_DIR / "task2_X_train.npy", mmap_mode="r", allow_pickle=False
    )
    y_train = np.load(DATA_DIR / "task2_y_train.npy", allow_pickle=False)
    X_test_mmap = np.load(
        DATA_DIR / "task2_X_test.npy", mmap_mode="r", allow_pickle=False
    )
    y_test = np.load(DATA_DIR / "task2_y_test.npy", allow_pickle=False)

    print(f"Train: {X_train_mmap.shape}  |  Test: {X_test_mmap.shape}")

    # ── 2. Build MobileNetV2 feature extractor ────────────────────────
    print("\nBuilding MobileNetV2 feature extractor...")
    extractor = build_mobilenet_extractor()

    # ── 3. Steady-state runtime pilot ─────────────────────────────────
    run_runtime_pilot(extractor, X_train_mmap)

    # ── 4. Extract and cache features (Resumable & Disk-Backed) ────────
    train_cache = CACHE_DIR / "train_features_mobilenet_v2.npy"
    test_cache = CACHE_DIR / "test_features_mobilenet_v2.npy"

    X_train_feat = extract_and_cache(extractor, X_train_mmap, train_cache, batch_size=8)
    X_test_feat = extract_and_cache(extractor, X_test_mmap, test_cache, batch_size=8)

    # ── 5. Stratified 80/20 train/validation split ────────────────────
    y_classes = np.argmax(y_train, axis=1)
    idx_tr, idx_va = train_test_split(
        np.arange(len(y_train)),
        test_size=0.2,
        random_state=42,
        stratify=y_classes,
    )
    print(f"\nTrain split: {len(idx_tr)} | Val split: {len(idx_va)}")

    # ── 6. Build and train classification head ────────────────────────
    head_input = tf.keras.Input(shape=(1280,))
    x = tf.keras.layers.Dropout(0.4)(head_input)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    head_out = tf.keras.layers.Dense(2, activation="softmax")(x)

    head_model = tf.keras.Model(head_input, head_out, name="ClassificationHead")
    head_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n--- Training Head on Cached Features (CPU) ---")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        ),
    ]

    history = head_model.fit(
        X_train_feat[idx_tr],
        y_train[idx_tr],
        validation_data=(X_train_feat[idx_va], y_train[idx_va]),
        epochs=25,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training history
    with open(METRICS_DIR / "training_history.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f,
            indent=2,
        )

    # ── 7. Plot training curves ───────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history["loss"], label="Train")
        ax1.plot(history.history["val_loss"], label="Validation")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(history.history["accuracy"], label="Train")
        ax2.plot(history.history["val_accuracy"], label="Validation")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "training_curves.png", dpi=150)
        plt.close()
        print(f"✅ Training curves saved to {PLOTS_DIR / 'training_curves.png'}")
    except Exception as e:
        print(f"⚠️  Could not save training plots: {e}")

    # ── 8. Evaluate on official held-out test set ─────────────────────
    print("\n--- Evaluating on Official Test Set (1,460 samples) ---")
    test_probs = head_model.predict(X_test_feat)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(test_probs, axis=1)

    acc = float(np.mean(y_true == y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # Class 0 is Damaged (positive class)
    auc = float(roc_auc_score((y_true == 0).astype(int), test_probs[:, 0]))

    print(f"\n{'='*50}")
    print(f"Official Test Accuracy: {acc * 100:.2f}%")
    print(f"ROC-AUC (Damaged):      {auc:.4f}")
    print(f"{'='*50}")
    print("\nConfusion Matrix:")
    print(f"              Pred Damaged  Pred Undamaged")
    print(f"  Damaged       {cm[0,0]:>6}         {cm[0,1]:>6}")
    print(f"  Undamaged     {cm[1,0]:>6}         {cm[1,1]:>6}")

    report_dict = classification_report(
        y_true, y_pred, target_names=["Damaged", "Undamaged"], output_dict=True
    )
    report_str = classification_report(
        y_true, y_pred, target_names=["Damaged", "Undamaged"]
    )
    print(f"\nClassification Report:\n{report_str}")

    # Save metrics
    with open(METRICS_DIR / "classification_report.txt", "w") as f:
        f.write(report_str)
    with open(METRICS_DIR / "test_metrics.json", "w") as f:
        json.dump(
            {
                "accuracy": acc,
                "roc_auc_damaged": auc,
                "confusion_matrix": cm.tolist(),
                "classification_report": report_dict,
            },
            f,
            indent=2,
        )
    print(f"✅ Metrics saved to {METRICS_DIR}")

    # ── 9. Plot confusion matrix and ROC ──────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Damaged", "Undamaged"])
        ax.set_yticklabels(["Damaged", "Undamaged"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve((y_true == 0).astype(int), test_probs[:, 0])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="darkorange", lw=2,
                label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (Damaged Class)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=150)
        plt.close()

        print(f"✅ Plots saved to {PLOTS_DIR}")
    except Exception as e:
        print(f"⚠️  Could not save evaluation plots: {e}")

    # ── 10. Assemble unified end-to-end model ─────────────────────────
    full_input = tf.keras.Input(shape=(224, 224, 3))
    extracted = extractor(full_input, training=False)
    final_probs = head_model(extracted)

    full_model = tf.keras.Model(
        full_input, final_probs, name="MobileNetV2_DamageClassifier"
    )
    final_path = MODELS_DIR / "mobilenet_v2_damage_classifier.keras"
    full_model.save(final_path)
    print(f"\n✅ Final unified model saved to: {final_path}")
    print("   Use this model in the Streamlit app for single-image inference.")


if __name__ == "__main__":
    main()
