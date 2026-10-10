from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

SEED = 42
RNG = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parent / "embedded" / "anfis_model.h"

X = RNG.uniform(-2.0, 2.0, size=(1200, 2)).astype(np.float32)
y = (np.sin(X[:, 0]) + 0.45 * X[:, 1] + 0.15 * X[:, 0] * X[:, 1]).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

centers = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
sigmas = np.array([[0.95, 0.95], [0.95, 0.95]], dtype=np.float32)


def firing(x):
    mu = np.empty((len(x), 2, 2), dtype=np.float64)
    for f in range(2):
        for m in range(2):
            mu[:, f, m] = np.exp(-0.5 * ((x[:, f] - centers[f, m]) / sigmas[f, m]) ** 2)
    w = np.stack([
        mu[:, 0, 0] * mu[:, 1, 0],
        mu[:, 0, 0] * mu[:, 1, 1],
        mu[:, 0, 1] * mu[:, 1, 0],
        mu[:, 0, 1] * mu[:, 1, 1],
    ], axis=1)
    return w / (w.sum(axis=1, keepdims=True) + 1e-12)

w_train = firing(X_train)
blocks = []
for r in range(4):
    blocks.append(w_train[:, [r]] * np.column_stack([X_train[:, 0], X_train[:, 1], np.ones(len(X_train))]))
A = np.concatenate(blocks, axis=1)
coef, *_ = np.linalg.lstsq(A, y_train, rcond=None)
coef = coef.reshape(4, 3)

w_test = firing(X_test)
y_pred = np.zeros(len(X_test), dtype=np.float64)
for r in range(4):
    y_pred += w_test[:, r] * (coef[r, 0] * X_test[:, 0] + coef[r, 1] * X_test[:, 1] + coef[r, 2])
print("MAE:", round(float(mean_absolute_error(y_test, y_pred)), 4))


def arr(name, values):
    flat = np.asarray(values).reshape(-1)
    return f"static const float {name}[{len(flat)}] = {{{', '.join(f'{float(v):.9f}f' for v in flat)}}};\n"

header = "#pragma once\n#include <math.h>\n\n"
header += arr("ANFIS_CENTERS", centers)
header += arr("ANFIS_SIGMAS", sigmas)
header += arr("ANFIS_COEF", coef)
header += r'''
inline float anfisMembership(float x, float c, float s) {
  float d = (x - c) / s;
  return expf(-0.5f * d * d);
}

inline float anfisPredict(const float *x) {
  float m00 = anfisMembership(x[0], ANFIS_CENTERS[0], ANFIS_SIGMAS[0]);
  float m01 = anfisMembership(x[0], ANFIS_CENTERS[1], ANFIS_SIGMAS[1]);
  float m10 = anfisMembership(x[1], ANFIS_CENTERS[2], ANFIS_SIGMAS[2]);
  float m11 = anfisMembership(x[1], ANFIS_CENTERS[3], ANFIS_SIGMAS[3]);
  float w[4] = {m00*m10, m00*m11, m01*m10, m01*m11};
  float sum = w[0] + w[1] + w[2] + w[3] + 1e-12f;
  float y = 0.0f;
  for (int r = 0; r < 4; ++r) {
    float f = ANFIS_COEF[r*3] * x[0] + ANFIS_COEF[r*3+1] * x[1] + ANFIS_COEF[r*3+2];
    y += (w[r] / sum) * f;
  }
  return y;
}
'''
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(header, encoding="utf-8")
print(f"Exported model to {OUT}")
