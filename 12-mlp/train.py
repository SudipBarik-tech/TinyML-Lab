from pathlib import Path
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

SEED = 42
OUT = Path(__file__).resolve().parent / "embedded" / "mlp_model.h"

X, y = make_moons(n_samples=500, noise=0.18, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(6,),
    activation="tanh",
    solver="lbfgs",
    alpha=1e-4,
    max_iter=2000,
    random_state=SEED,
)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)
print("Test accuracy:", round(accuracy_score(y_test, pred), 4))

w1 = model.coefs_[0]          # shape: 2 x 6
b1 = model.intercepts_[0]     # shape: 6
w2 = model.coefs_[1].reshape(-1)  # shape: 6
b2 = float(model.intercepts_[1][0])

fmt = lambda v: f"{float(v):.9f}f"

w1_rows = ",\n  ".join(
    "{" + ", ".join(fmt(v) for v in row) + "}" for row in w1
)

header = f'''#pragma once
#include <math.h>

static const int MLP_INPUTS = 2;
static const int MLP_HIDDEN = 6;
static const float MLP_MEAN[2] = {{{fmt(scaler.mean_[0])}, {fmt(scaler.mean_[1])}}};
static const float MLP_SCALE[2] = {{{fmt(scaler.scale_[0])}, {fmt(scaler.scale_[1])}}};
static const float MLP_W1[2][6] = {{
  {w1_rows}
}};
static const float MLP_B1[6] = {{{', '.join(fmt(v) for v in b1)}}};
static const float MLP_W2[6] = {{{', '.join(fmt(v) for v in w2)}}};
static const float MLP_B2 = {fmt(b2)};

inline float mlpProbability(const float *x) {{
  float z[MLP_INPUTS];
  for (int i = 0; i < MLP_INPUTS; ++i) z[i] = (x[i] - MLP_MEAN[i]) / MLP_SCALE[i];

  float h[MLP_HIDDEN];
  for (int j = 0; j < MLP_HIDDEN; ++j) {{
    float s = MLP_B1[j];
    for (int i = 0; i < MLP_INPUTS; ++i) s += z[i] * MLP_W1[i][j];
    h[j] = tanhf(s);
  }}

  float out = MLP_B2;
  for (int j = 0; j < MLP_HIDDEN; ++j) out += h[j] * MLP_W2[j];
  return 1.0f / (1.0f + expf(-out));
}}

inline int mlpPredict(const float *x) {{
  return mlpProbability(x) >= 0.5f ? 1 : 0;
}}
'''

OUT.write_text(header, encoding="utf-8")
print(f"Exported embedded model to {OUT}")
