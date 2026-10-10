from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path(__file__).resolve().parent / "embedded" / "rbf_model.h"

X, y = make_moons(n_samples=800, noise=0.20, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

n_centres = 10
kmeans = KMeans(n_clusters=n_centres, random_state=SEED, n_init=10)
kmeans.fit(X_train_s)
centres = kmeans.cluster_centers_

centre_distances = []
for i in range(n_centres):
    for j in range(i + 1, n_centres):
        centre_distances.append(np.linalg.norm(centres[i] - centres[j]))
sigma = max(float(np.median(centre_distances)), 1e-3)
gamma = 1.0 / (2.0 * sigma * sigma)


def rbf_features(x):
    d2 = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * d2)

Phi_train = rbf_features(X_train_s)
Phi_test = rbf_features(X_test_s)

model = LogisticRegression(max_iter=1000, random_state=SEED)
model.fit(Phi_train, y_train)
pred = model.predict(Phi_test)
print("Accuracy:", round(float(accuracy_score(y_test, pred)), 4))

weights = model.coef_.reshape(-1)
bias = float(model.intercept_[0])


def c_array(name, arr):
    flat = np.asarray(arr).reshape(-1)
    vals = ", ".join(f"{float(v):.9f}f" for v in flat)
    return f"const float {name}[{flat.size}] = {{{vals}}};\n"

header = "#pragma once\n#include <math.h>\n\n"
header += "const int RBF_FEATURES = 2;\n"
header += f"const int RBF_CENTRES = {n_centres};\n"
header += f"const float RBF_GAMMA = {gamma:.9f}f;\n"
header += c_array("RBF_MEAN", scaler.mean_)
header += c_array("RBF_SCALE", scaler.scale_)
header += c_array("RBF_CENTRE_DATA", centres)
header += c_array("RBF_WEIGHTS", weights)
header += f"const float RBF_BIAS = {bias:.9f}f;\n"
header += r'''
inline float rbfScore(const float *x) {
  float z[RBF_FEATURES];
  for (int j = 0; j < RBF_FEATURES; ++j) {
    z[j] = (x[j] - RBF_MEAN[j]) / RBF_SCALE[j];
  }

  float score = RBF_BIAS;
  for (int c = 0; c < RBF_CENTRES; ++c) {
    float d2 = 0.0f;
    for (int j = 0; j < RBF_FEATURES; ++j) {
      float d = z[j] - RBF_CENTRE_DATA[c * RBF_FEATURES + j];
      d2 += d * d;
    }
    score += RBF_WEIGHTS[c] * expf(-RBF_GAMMA * d2);
  }
  return score;
}

inline int rbfPredict(const float *x) {
  return rbfScore(x) >= 0.0f ? 1 : 0;
}
'''
OUT.write_text(header, encoding="utf-8")
print(f"Exported embedded model to {OUT}")
