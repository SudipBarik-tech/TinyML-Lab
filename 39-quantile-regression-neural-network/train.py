from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42
RNG = np.random.default_rng(SEED)
tf.keras.utils.set_random_seed(SEED)
OUT = Path(__file__).resolve().parent / "embedded" / "qrnn_model.h"

QUANTILES = np.array([0.10, 0.50, 0.90], dtype=np.float32)
HIDDEN = 12

X = RNG.uniform(-2.0, 2.0, size=(1800, 2)).astype(np.float32)
scale = 0.15 + 0.35 * (np.abs(X[:, 0]) / 2.0)
noise = RNG.normal(0.0, scale)
y = (1.5 * X[:, 0] - 0.7 * X[:, 1] + 0.25 * X[:, 0] * X[:, 1] + noise).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(HIDDEN, activation="relu"),
    tf.keras.layers.Dense(len(QUANTILES)),
])

q = tf.constant(QUANTILES.reshape(1, -1), dtype=tf.float32)

def quantile_loss(y_true, y_pred):
    e = tf.expand_dims(y_true, axis=-1) - y_pred
    return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=quantile_loss)
model.fit(X_train, y_train, epochs=80, batch_size=32, verbose=0)

pred = model.predict(X_test, verbose=0)
coverage_10 = np.mean(y_test <= pred[:, 0])
coverage_50 = np.mean(y_test <= pred[:, 1])
coverage_90 = np.mean(y_test <= pred[:, 2])
crossing = np.mean((pred[:, 0] > pred[:, 1]) | (pred[:, 1] > pred[:, 2]))
print("Empirical coverage q10/q50/q90:", *(round(float(v), 3) for v in [coverage_10, coverage_50, coverage_90]))
print("Quantile crossing rate:", round(float(crossing), 4))

w1, b1 = model.layers[0].get_weights()
w2, b2 = model.layers[1].get_weights()


def arr(name, values):
    flat = np.asarray(values).reshape(-1)
    return f"static const float {name}[{len(flat)}] = {{{', '.join(f'{float(v):.9f}f' for v in flat)}}};\n"

header = "#pragma once\n\n"
header += f"static const int QRNN_INPUTS = 2;\nstatic const int QRNN_HIDDEN = {HIDDEN};\nstatic const int QRNN_OUTPUTS = 3;\n"
header += arr("QRNN_W1", w1)
header += arr("QRNN_B1", b1)
header += arr("QRNN_W2", w2)
header += arr("QRNN_B2", b2)
header += r'''
inline void qrnnPredict(const float *x, float *out) {
  float h[QRNN_HIDDEN];
  for (int j = 0; j < QRNN_HIDDEN; ++j) {
    float s = QRNN_B1[j];
    for (int i = 0; i < QRNN_INPUTS; ++i) s += x[i] * QRNN_W1[i * QRNN_HIDDEN + j];
    h[j] = s > 0.0f ? s : 0.0f;
  }
  for (int j = 0; j < QRNN_OUTPUTS; ++j) {
    float s = QRNN_B2[j];
    for (int i = 0; i < QRNN_HIDDEN; ++i) s += h[i] * QRNN_W2[i * QRNN_OUTPUTS + j];
    out[j] = s;
  }
}
'''
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(header, encoding="utf-8")
print(f"Exported model to {OUT}")
