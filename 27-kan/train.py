from pathlib import Path
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path(__file__).resolve().parent / "embedded" / "kan_model.h"

X, y = make_moons(n_samples=700, noise=0.18, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

centres = np.linspace(-2.5, 2.5, 7, dtype=np.float64)
gamma = 1.25


def basis_expand(x):
    # Independent univariate basis functions for each input feature.
    parts = []
    for j in range(x.shape[1]):
        diff = x[:, j:j+1] - centres.reshape(1, -1)
        parts.append(np.exp(-gamma * diff * diff))
    return np.concatenate(parts, axis=1)

Phi_train = basis_expand(X_train_s)
Phi_test = basis_expand(X_test_s)

model = LogisticRegression(C=5.0, max_iter=1000, random_state=SEED)
model.fit(Phi_train, y_train)
pred = model.predict(Phi_test)
print("Accuracy:", round(float(accuracy_score(y_test, pred)), 4))

weights = model.coef_.reshape(-1)
bias = float(model.intercept_[0])


def c_array(name, arr):
    vals = ", ".join(f"{float(v):.9f}f" for v in np.asarray(arr).reshape(-1))
    return f"const float {name}[{np.asarray(arr).size}] = {{{vals}}};\n"

header = "#pragma once\n#include <math.h>\n\n"
header += "const int KAN_FEATURES = 2;\n"
header += f"const int KAN_BASIS = {len(centres)};\n"
header += f"const float KAN_GAMMA = {gamma:.9f}f;\n"
header += c_array("KAN_MEAN", scaler.mean_)
header += c_array("KAN_SCALE", scaler.scale_)
header += c_array("KAN_CENTRES", centres)
header += c_array("KAN_WEIGHTS", weights)
header += f"const float KAN_BIAS = {bias:.9f}f;\n"
header += r'''
inline float kanScore(const float *x) {
  float score = KAN_BIAS;
  int w = 0;
  for (int j = 0; j < KAN_FEATURES; ++j) {
    float z = (x[j] - KAN_MEAN[j]) / KAN_SCALE[j];
    for (int k = 0; k < KAN_BASIS; ++k) {
      float d = z - KAN_CENTRES[k];
      float phi = expf(-KAN_GAMMA * d * d);
      score += KAN_WEIGHTS[w++] * phi;
    }
  }
  return score;
}

inline int kanPredict(const float *x) {
  return kanScore(x) >= 0.0f ? 1 : 0;
}
'''
OUT.write_text(header, encoding="utf-8")
print(f"Exported embedded model to {OUT}")
