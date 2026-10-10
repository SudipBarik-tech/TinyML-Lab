from pathlib import Path
import numpy as np
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
OUT = Path(__file__).resolve().parent / "embedded" / "micro_mobilenet_model.h"

# Create a small synthetic 16x16 grayscale classification dataset.
rng = np.random.default_rng(SEED)
n = 1200
X = rng.normal(0, 0.15, size=(n, 16, 16, 1)).astype(np.float32)
y = rng.integers(0, 3, size=n)

for i, cls in enumerate(y):
    if cls == 0:
        X[i, 3:13, 7:9, 0] += 1.0
    elif cls == 1:
        X[i, 7:9, 3:13, 0] += 1.0
    else:
        for d in range(3, 13):
            X[i, d, d, 0] += 1.0

idx = rng.permutation(n)
train_idx, test_idx = idx[:900], idx[900:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

inputs = tf.keras.Input(shape=(16, 16, 1))
x = tf.keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu")(inputs)
x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
x = tf.keras.layers.Conv2D(12, 1, activation="relu")(x)
x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", activation="relu")(x)
x = tf.keras.layers.Conv2D(16, 1, activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=18, batch_size=32, verbose=0)
_, acc = model.evaluate(X_test, y_test, verbose=0)
print("Validation accuracy:", round(float(acc), 4))
print("Parameters:", model.count_params())


def representative_data():
    for i in range(min(200, len(X_train))):
        yield [X_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
print("INT8 model size (bytes):", len(tflite_model))

hex_bytes = ", ".join(f"0x{b:02x}" for b in tflite_model)
OUT.write_text(
    f"#pragma once\n\nconst unsigned char MICRO_MOBILENET_MODEL[] = {{{hex_bytes}}};\n"
    f"const unsigned int MICRO_MOBILENET_MODEL_LEN = {len(tflite_model)};\n",
    encoding="utf-8",
)
print(f"Exported model to {OUT}")
