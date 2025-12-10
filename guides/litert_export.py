"""
Title: Exporting Keras models to LiteRT (TensorFlow Lite)
Author: [Rahul Kumar](https://github.com/rahulkumar-aws)
Date created: 2025/12/10
Last modified: 2025/12/10
Description: Complete guide to exporting Keras models for mobile and edge deployment.
Accelerator: None
"""

"""
## Introduction

LiteRT (formerly TensorFlow Lite) enables you to deploy Keras models on mobile,
embedded, and edge devices. This guide covers the **one-line export API** in Keras 3.x
that makes mobile deployment simple.

### What you'll learn

- Export Keras models to `.tflite` format with a single line of code
- Work with different model types (Sequential, Functional, Subclassed)
- Export Keras-Hub pretrained models
- Apply quantization for smaller model sizes
- Handle dynamic input shapes

### Key benefits

- **One-line export**: `model.export("model.tflite", format="litert")`
- **Multi-backend support**: Train with JAX/PyTorch, export to LiteRT
- **Automatic input handling**: Works with dict inputs (Keras-Hub models)
- **Built-in optimization**: Quantization support via `litert_kwargs`
"""

"""
## Setup
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
from keras import layers
import tensorflow as tf

"""
## Quick Start: Export a Simple Model

Let's start with a basic Sequential model and export it to LiteRT format.
"""

# Create a simple model
model = keras.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Export to LiteRT (one line!)
model.export("mnist_classifier.tflite", format="litert")

print("Exported mnist_classifier.tflite")

"""
That's it! The model is now ready for mobile deployment.

Let's verify the exported model works correctly.
"""

# Load and test the exported model
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

interpreter = Interpreter(model_path="mnist_classifier.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nModel Input Details:")
print(f"  Shape: {input_details[0]['shape']}")
print(f"  Type: {input_details[0]['dtype']}")

print("\nModel Output Details:")
print(f"  Shape: {output_details[0]['shape']}")
print(f"  Type: {output_details[0]['dtype']}")

# Test inference
test_input = np.random.random((1, 784)).astype(np.float32)
interpreter.set_tensor(input_details[0]["index"], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]["index"])

print(f"\nInference successful! Output shape: {output.shape}")

"""
## Exporting Different Model Types

### Functional Models

Functional models are the recommended way to build Keras models.
"""

# Create a Functional model
inputs = keras.Input(shape=(224, 224, 3), name="image_input")
x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

functional_model = keras.Model(inputs=inputs, outputs=outputs, name="image_classifier")

# Export to LiteRT
functional_model.export("image_classifier.tflite", format="litert")

print("Exported Functional model")

"""
### Subclassed Models

For subclassed models, you must call the model with sample data first to establish
input shapes before exporting.
"""


class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(128, activation="relu")
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(10, activation="softmax")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)


# Create and initialize the model
custom_model = CustomModel()

# IMPORTANT: Call model with sample data to establish input shapes
sample_input = np.zeros((1, 784), dtype=np.float32)
_ = custom_model(sample_input, training=False)

# Now export
custom_model.export("custom_model.tflite", format="litert")

print("Exported Subclassed model")

"""
## Exporting Keras-Hub Models

Keras-Hub models work seamlessly with the export API. Input configurations like
sequence length are set during model construction via the preprocessor.
"""

"""shell
pip install keras-hub
"""

import keras_hub

# Load a pretrained text model
# Sequence length is configured via the preprocessor
preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(
    "gemma_2b_en", sequence_length=128
)

model = keras_hub.models.GemmaCausalLM.from_preset(
    "gemma_2b_en", preprocessor=preprocessor
)

# Export to LiteRT (sequence length already set)
model.export("gemma_2b.tflite", format="litert")

print("Exported Keras-Hub Gemma model")

"""
For vision models, the image size is determined by the preset:
"""

# Load a vision model
vision_model = keras_hub.models.ImageClassifier.from_preset(
    "efficientnetv2_b0_imagenet"
)

# Export (image size already set by preset)
vision_model.export("efficientnet.tflite", format="litert")

print("Exported Keras-Hub vision model")

"""
## Quantization for Smaller Models

Quantization reduces model size and improves inference speed by using lower-precision
data types. This is crucial for mobile deployment.

### Dynamic Range Quantization

The simplest quantization method converts weights from float32 to int8.
"""

# Create a model
quantization_model = keras.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=(784,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Export with dynamic range quantization
quantization_model.export(
    "model_quantized.tflite",
    format="litert",
    litert_kwargs={"optimizations": [tf.lite.Optimize.DEFAULT]},
)

print("Exported quantized model")

# Compare file sizes
import os

original_size = os.path.getsize("mnist_classifier.tflite") / 1024
quantized_size = os.path.getsize("model_quantized.tflite") / 1024

print(f"\nOriginal model: {original_size:.2f} KB")
print(f"Quantized model: {quantized_size:.2f} KB")
print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")

"""
### Float16 Quantization

Float16 quantization provides a good balance between model size and accuracy,
especially for GPU inference.
"""

# Export with float16 quantization
quantization_model.export(
    "model_float16.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "target_spec": {"supported_types": [tf.float16]},
    },
)

print("Exported Float16 quantized model")

"""
### Full Integer Quantization (INT8)

For maximum optimization, use full integer quantization with a representative dataset.
This quantizes both weights and activations.
"""


# Prepare a representative dataset for calibration
def representative_dataset():
    """Generate calibration data from your validation set."""
    for _ in range(100):
        # Use real validation data for best results
        sample = np.random.random((1, 784)).astype(np.float32)
        yield [sample]


# Export with INT8 quantization using TFLite converter directly
converter = tf.lite.TFLiteConverter.from_keras_model(quantization_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()

with open("model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("Exported INT8 quantized model")

int8_size = os.path.getsize("model_int8.tflite") / 1024
print(f"INT8 model size: {int8_size:.2f} KB")

"""
## Dynamic Input Shapes

Export models with flexible input dimensions that can be resized at runtime.
This is useful for variable-length sequences or different image sizes.
"""

# Create a model with dynamic sequence length
dynamic_model = keras.Sequential(
    [
        layers.Input(shape=(None,)),  # None = dynamic dimension
        layers.Embedding(input_dim=1000, output_dim=64),
        layers.GlobalAveragePooling1D(),
        layers.Dense(10, activation="softmax"),
    ]
)

# Export with dynamic shapes
dynamic_model.export("dynamic_model.tflite", format="litert")

print("Exported model with dynamic shapes")

# Verify dynamic shapes in the exported model
interpreter = Interpreter(model_path="dynamic_model.tflite")
input_details = interpreter.get_input_details()

print(f"\nInput shape: {input_details[0]['shape']}")
print("Note: -1 indicates a dynamic dimension")

"""
## Custom Input Signatures

For advanced use cases, you can specify custom input signatures using TensorSpec.
"""

# Create a model expecting multiple inputs
input1 = keras.Input(shape=(32,), name="input1")
input2 = keras.Input(shape=(64,), name="input2")

x1 = layers.Dense(64)(input1)
x2 = layers.Dense(64)(input2)

combined = layers.Concatenate()([x1, x2])
outputs = layers.Dense(10, activation="softmax")(combined)

multi_input_model = keras.Model(inputs=[input1, input2], outputs=outputs)

# Export with custom input signature
multi_input_model.export(
    "multi_input.tflite",
    format="litert",
    input_signature=[
        tf.TensorSpec(shape=(None, 32), dtype=tf.float32, name="input1"),
        tf.TensorSpec(shape=(None, 64), dtype=tf.float32, name="input2"),
    ],
)

print("Exported multi-input model with custom signature")

"""
## Cross-Backend Export

Keras 3.x supports multiple backends (JAX, PyTorch, TensorFlow). You can train
with any backend and export to LiteRT using TensorFlow backend.

### Training with JAX, Exporting with TensorFlow

Here's a typical workflow:
"""

# Simulate training with JAX backend (in a separate script)
# os.environ["KERAS_BACKEND"] = "jax"
# import keras
# model = keras.Sequential([...])
# model.fit(X_train, y_train)
# model.save_weights("model_weights.weights.h5")

# Export script (use TensorFlow backend)
os.environ["KERAS_BACKEND"] = "tensorflow"

# Recreate model architecture
export_model = keras.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=(784,)),
        layers.Dense(10, activation="softmax"),
    ]
)

# Load weights trained with JAX (backend-agnostic weights)
# export_model.load_weights("model_weights.weights.h5")

# Export to LiteRT
export_model.export("cross_backend_model.tflite", format="litert")

print("Cross-backend export demonstrated")

"""
## Validation Best Practices

Always verify your exported model before deploying to production.
"""


def validate_tflite_model(model_path, keras_model):
    """Compare TFLite model output with Keras model."""
    # Load TFLite model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create test input
    test_input = np.random.random(input_details[0]["shape"]).astype(np.float32)

    # Get Keras prediction
    keras_output = keras_model.predict(test_input, verbose=0)

    # Get TFLite prediction
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]["index"])

    # Calculate difference
    diff = np.abs(keras_output - tflite_output).mean()

    print(f"\nValidation Results:")
    print(f"  Mean absolute difference: {diff:.6f}")

    if diff < 1e-4:
        print("  Outputs match (excellent)")
    elif diff < 1e-2:
        print("  Small difference (acceptable for quantized models)")
    else:
        print("  Large difference (investigate!)")

    return diff


# Validate the exported model
validate_tflite_model("mnist_classifier.tflite", model)

"""
## Troubleshooting Common Issues

### Backend Error

**Error**: `RuntimeError: Backend must be TensorFlow`

**Solution**: Set the TensorFlow backend before importing Keras:

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
```

### Subclassed Model Export Fails

**Error**: Unable to infer input signature for subclassed model

**Solution**: Call the model with sample data before exporting:

```python
sample_input = np.zeros((1, 784), dtype=np.float32)
_ = model(sample_input, training=False)
model.export("model.tflite", format="litert")
```

### Unsupported Operations

**Error**: Some ops are not supported by TFLite

**Solution**: Enable TensorFlow ops in LiteRT:

```python
model.export(
    "model.tflite",
    format="litert",
    litert_kwargs={
        "target_spec": {
            "supported_ops": [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
        }
    }
)
```
"""

"""
## Summary

In this guide, you learned how to:

- Export Keras models to LiteRT format with `model.export()`
- Work with Sequential, Functional, and Subclassed models
- Export Keras-Hub pretrained models
- Apply quantization to reduce model size (up to 75% reduction)
- Handle dynamic input shapes for flexible inference
- Validate exported models before deployment

### Key Takeaways

- **One-line export**: `model.export("model.tflite", format="litert")`
- **Use quantization**: Reduce size by 50-75% with minimal accuracy loss
- **Always validate**: Test exported models before production deployment
- **Backend flexibility**: Train with JAX/PyTorch, export with TensorFlow

### Next Steps

- Deploy your `.tflite` model to mobile apps (Android/iOS)
- Use GPU delegates for faster inference on mobile devices
- Explore model optimization techniques for specific hardware
- Check out the [Keras-Hub documentation](https://keras.io/keras_hub/) for
  pretrained models

For more details, see:
- [Keras Serialization Guide](https://keras.io/guides/serialization_and_saving/)
- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [Quantization in Keras Guide](https://keras.io/guides/quantization_overview/)
"""
