from pathlib import Path
import numpy as np
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
OUT = Path(__file__).resolve().parent / "embedded" / "autoencoder_model.h"

rng = np.random.default_rng(SEED)
N, FEATURES = 1200, 6
latent = rng.normal(size=(N, 2)).astype(np.float32)
W = np.array([
    [1.0, 0.2], [0.8, -0.5], [0.3, 1.1],
    [-0.6, 0.7], [0.5, 0.6], [-0.4, -0.8]
], dtype=np.float32)
X = latent @ W.T + rng.normal(0, 0.08, size=(N, FEATURES)).astype(np.float32)
mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-6
Xs = (X - mean) / std

cut = int(0.8 * N)
X_train, X_test = Xs[:cut], Xs[cut:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(FEATURES,)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(2, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(FEATURES),
])
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, X_train, epochs=20, batch_size=32, verbose=0, validation_split=0.1)

train_rec = model.predict(X_train, verbose=0)
train_err = np.mean((X_train - train_rec) ** 2, axis=1)
threshold = float(np.percentile(train_err, 99))

test_rec = model.predict(X_test, verbose=0)
test_err = np.mean((X_test - test_rec) ** 2, axis=1)
print("Normal test mean error:", round(float(test_err.mean()), 6))
print("Anomaly threshold:", round(threshold, 6))

anom = rng.normal(3.0, 1.0, size=(150, FEATURES)).astype(np.float32)
anom = (anom - mean) / std
anom_rec = model.predict(anom, verbose=0)
anom_err = np.mean((anom - anom_rec) ** 2, axis=1)
print("Detected anomalies:", round(float(np.mean(anom_err > threshold)), 4))

def representative_dataset():
    for sample in X_train[:200]:
        yield [sample[np.newaxis, :].astype(np.float32)]

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
    f"const unsigned char AUTOENCODER_MODEL[] = {{{hex_bytes}}};\n"
    f"const unsigned int AUTOENCODER_MODEL_LEN = {len(tflite_model)};\n"
    f"constexpr int AE_FEATURES = {FEATURES};\n"
    f"constexpr float AE_THRESHOLD = {threshold:.9f}f;\n"
    "const float AE_MEAN[AE_FEATURES] = {" + ", ".join(f"{v:.9f}f" for v in mean) + "};\n"
    "const float AE_STD[AE_FEATURES] = {" + ", ".join(f"{v:.9f}f" for v in std) + "};\n"
)
OUT.write_text(header, encoding="utf-8")
print(f"Exported model to {OUT}")
