"""Streamlit application for Structural Damage Recognition.

Accepts uploaded JPG/PNG images and predicts whether the structure
shown is Damaged or Undamaged using the trained MobileNetV2 transfer learning model.
"""

import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

LABEL_NAMES = {0: "Damaged", 1: "Undamaged"}


@st.cache_resource(show_spinner="Loading trained damage recognition model...")
def load_cached_model():
    """Load the trained Keras model with caching to avoid repeated loads."""
    import tensorflow as tf

    model_paths = [
        Path("models/mobilenet_v2_damage_classifier.keras"),
        Path("models/best_model.keras"),
    ]

    for model_path in model_paths:
        if model_path.exists():
            try:
                model = tf.keras.models.load_model(str(model_path), compile=False)
                return model, model_path.name
            except Exception as e:
                st.error(f"Error loading {model_path}: {e}")

    return None, None


def preprocess_for_inference(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """Prepare an uploaded RGB image for inference.

    CRITICAL PREPROCESSING NOTE:
    Ordinary uploaded photographs (JPG/PNG) are standard RGB [0, 255] images.
    They must NOT be processed with Caffe BGR mean addition (which was used only to
    restore preprocessed dataset arrays).
    For MobileNetV2 and InceptionV3 inference, the input RGB pixels in [0, 255]
    are scaled to [-1, 1] via: (img / 127.5) - 1.0.
    """
    # Resize with high-quality resampling if dimensions differ
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to numpy float32 array
    img_array = np.array(image, dtype=np.float32)

    # Ensure exactly 3 channels (RGB)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    elif img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)

    # Normalization to [-1.0, 1.0] matching training feature extractor
    img_array = (img_array / 127.5) - 1.0

    # Expand batch dimension -> (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)


def main():
    st.set_page_config(
        page_title="Structural Damage Recognition",
        page_icon="🏗️",
        layout="centered",
    )

    st.title("🏗️ Structural Damage Recognition")
    st.markdown(
        "Upload a photograph of a building or structural component to classify whether "
        "it exhibits **Damaged** or **Undamaged** state, powered by a transfer learning "
        "model trained on the benchmark [PEER Hub ImageNet (PHI-Net)](https://apps.peer.berkeley.edu/phi-net/) dataset."
    )

    st.divider()

    # Load cached model
    model, model_filename = load_cached_model()

    if model is not None:
        st.success(f"✅ Loaded active model: `{model_filename}`")
    else:
        st.warning(
            "⚠️ **No trained model found.**\n\n"
            "Please ensure `mobilenet_v2_damage_classifier.keras` is present in `models/`."
        )

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a structural photograph (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload an image showing a structural facade, column, beam, joint, or wall.",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("❌ Could not read the uploaded file as a valid image.")
            return

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
            st.caption(f"Input Resolution: {image.size[0]} × {image.size[1]} px")

        with col2:
            st.subheader("Classification Result")

            if model is not None:
                with st.spinner("Analyzing structural features..."):
                    input_tensor = preprocess_for_inference(image)
                    # predict_on_batch avoids Keras tf.function retracing overhead on single calls
                    predictions = model.predict_on_batch(input_tensor)
                    if hasattr(predictions, "numpy"):
                        predictions = predictions.numpy()

                prob_damaged = float(predictions[0][0])
                prob_undamaged = float(predictions[0][1])
                predicted_class = int(np.argmax(predictions[0]))
                predicted_label = LABEL_NAMES[predicted_class]
                confidence = float(predictions[0][predicted_class])

                if predicted_class == 0:
                    st.error(f"### 🔴 {predicted_label}")
                else:
                    st.success(f"### 🟢 {predicted_label}")

                st.metric(label="Model Confidence", value=f"{confidence * 100:.2f}%")

                st.markdown("**Class Probabilities:**")
                st.progress(prob_damaged, text=f"Damaged: {prob_damaged * 100:.2f}%")
                st.progress(prob_undamaged, text=f"Undamaged: {prob_undamaged * 100:.2f}%")

                st.caption(
                    f"Raw Logits/Probabilities: Damaged={prob_damaged:.4f}, Undamaged={prob_undamaged:.4f}"
                )
            else:
                st.info("Model unavailable. Please verify model weights file.")

    # Educational & Ethical Disclaimer
    st.divider()
    st.info(
        "ℹ️ **Disclaimer & Limitations:**\n"
        "- This application is an educational prototype developed for a Machine Learning mini-project (UE24CS352A).\n"
        "- Predictions are generated based on visual patterns in the PHI-Net dataset (224×224 resolution) and do **not** constitute a formal structural engineering safety certification.\n"
        "- Actual field inspections require licensed civil and structural engineers, multi-angle assessments, and material testing."
    )


if __name__ == "__main__":
    main()
