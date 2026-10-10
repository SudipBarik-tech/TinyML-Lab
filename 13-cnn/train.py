from pathlib import Path
import numpy as np
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE = Path(__file__).resolve().parent
TFLITE_PATH = BASE / "tiny_cnn_int8.tflite"
HEADER_PATH = BASE / "embedded" / "cnn_model_data.h"

N = 800
L = 64
X = np.zeros((N, L, 1), dtype=np.float32)
y = np.zeros((N,), dtype=np.int32)
t = np.linspace(0, 1, L, endpoint=False, dtype=np.float32)

rng = np.random.default_rng(SEED)
for i in range(N):
    label = i % 2
    freq = 3.0 if label == 0 else 7.0
    phase = rng.uniform(0, 2 * np.pi)
    signal = np.sin(2 * np.pi * freq * t + phase)
    noise = rng.normal(0, 0.15, size=L)
    X[i, :, 0] = (signal + noise).astype(np.float32)
    y[i] = label

idx = rng.permutation(N)
X, y = X[idx], y[idx]
split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(L, 1)),
    tf.keras.layers.Conv1D(6, 5, activation="relu", padding="same"),
    tf.keras.layers.MaxPool1D(2),
    tf.keras.layers.Conv1D(8, 3, activation="relu", padding="same"),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=12, batch_size=32, validation_split=0.15, verbose=0)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", round(float(acc), 4))


def representative_dataset():
    for i in range(120):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
TFLITE_PATH.write_bytes(tflite_model)
print("INT8 model size (bytes):", len(tflite_model))

hex_bytes = ", ".join(f"0x{b:02x}" for b in tflite_model)
header = f'''#pragma once
#include <stdint.h>

const unsigned char CNN_MODEL_DATA[] = {{
  {hex_bytes}
}};
const unsigned int CNN_MODEL_DATA_LEN = {len(tflite_model)};
'''
HEADER_PATH.write_text(header, encoding="utf-8")
print(f"Exported model header to {HEADER_PATH}")
