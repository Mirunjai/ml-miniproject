"""Model definitions for structural damage classification."""

import tensorflow as tf

def build_inception_model(input_shape=(224, 224, 3), num_classes=2, freeze_backbone=True, dropout_rate=0.5) -> tf.keras.Model:
    """Build InceptionV3-based transfer learning model."""
    base_model = tf.keras.applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    base_model.trainable = not freeze_backbone
    
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=not freeze_backbone)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs, name="InceptionV3_Damage")
    return model

def unfreeze_top_layers(model: tf.keras.Model, num_layers: int = 30) -> None:
    """Unfreeze the top layers of the backbone for fine-tuning."""
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break
            
    if backbone is None:
        raise ValueError("Could not find backbone model")
        
    backbone.trainable = True
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False

def get_model_summary_string(model: tf.keras.Model) -> str:
    """Get model summary as a string."""
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()
