from pathlib import Path
import numpy as np
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
OUT = Path(__file__).resolve().parent / "embedded" / "lstm_model.h"

N, STEPS, FEATURES, CLASSES = 900, 20, 3, 3
X = np.zeros((N, STEPS, FEATURES), dtype=np.float32)
y = np.zeros(N, dtype=np.int32)
rng = np.random.default_rng(SEED)
t = np.linspace(0, 2 * np.pi, STEPS)
for i in range(N):
    cls = i % CLASSES
    y[i] = cls
    phase = rng.uniform(0, np.pi)
    base = np.sin((cls + 1) * t + phase)
    X[i, :, 0] = base + rng.normal(0, 0.08, STEPS)
    X[i, :, 1] = np.cos((cls + 1) * t + phase) + rng.normal(0, 0.08, STEPS)
    X[i, :, 2] = (cls + 1) * 0.25 + rng.normal(0, 0.05, STEPS)

idx = rng.permutation(N)
cut = int(0.8 * N)
tr, te = idx[:cut], idx[cut:]
X_train, X_test = X[tr], X[te]
y_train, y_test = y[tr], y[te]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(STEPS, FEATURES)),
    # The sequence length is fixed, so unrolling avoids dynamic TensorList
    # operations that are difficult to lower to a built-in-only TFLite model.
    tf.keras.layers.LSTM(12, unroll=True),
    tf.keras.layers.Dense(CLASSES, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", round(float(acc), 4))


def representative_dataset():
    for sample in X_train[:150]:
        yield [sample[np.newaxis, ...].astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
print("INT8 model bytes:", len(tflite_model))

hex_bytes = ", ".join(f"0x{b:02x}" for b in tflite_model)
header = (
    "#pragma once\n\n"
    f"const unsigned char LSTM_MODEL[] = {{{hex_bytes}}};\n"
    f"const unsigned int LSTM_MODEL_LEN = {len(tflite_model)};\n"
    f"constexpr int LSTM_STEPS = {STEPS};\n"
    f"constexpr int LSTM_FEATURES = {FEATURES};\n"
    f"constexpr int LSTM_CLASSES = {CLASSES};\n"
)
OUT.write_text(header, encoding="utf-8")
print(f"Exported model to {OUT}")
