from pathlib import Path
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

SEED = 42
OUT = Path(__file__).resolve().parent / "embedded" / "gmm_model.h"

X, _ = make_blobs(
    n_samples=240,
    centers=[(-1.8, -1.0), (1.7, 1.4)],
    cluster_std=[0.65, 0.75],
    random_state=SEED,
)

model = GaussianMixture(
    n_components=2,
    covariance_type="diag",
    random_state=SEED,
    n_init=5,
)
model.fit(X)
labels = model.predict(X)

print("Weights:", np.round(model.weights_, 4))
print("Means:\n", np.round(model.means_, 4))
print("Diagonal variances:\n", np.round(model.covariances_, 4))
print("Cluster counts:", np.bincount(labels))

weights = model.weights_
means = model.means_
vars_ = model.covariances_

header = f'''#pragma once
#include <math.h>

static const int GMM_COMPONENTS = 2;
static const int GMM_FEATURES = 2;
static const float GMM_WEIGHTS[2] = {{{weights[0]:.9f}f, {weights[1]:.9f}f}};
static const float GMM_MEANS[2][2] = {{
  {{{means[0,0]:.9f}f, {means[0,1]:.9f}f}},
  {{{means[1,0]:.9f}f, {means[1,1]:.9f}f}}
}};
static const float GMM_VARS[2][2] = {{
  {{{vars_[0,0]:.9f}f, {vars_[0,1]:.9f}f}},
  {{{vars_[1,0]:.9f}f, {vars_[1,1]:.9f}f}}
}};

inline int gmmPredict(const float *x) {{
  int best = 0;
  float bestScore = -1.0e30f;
  const float log2pi = 1.8378770664f;
  for (int k = 0; k < GMM_COMPONENTS; ++k) {{
    float score = logf(GMM_WEIGHTS[k]);
    for (int j = 0; j < GMM_FEATURES; ++j) {{
      const float d = x[j] - GMM_MEANS[k][j];
      score += -0.5f * (log2pi + logf(GMM_VARS[k][j]) + (d * d) / GMM_VARS[k][j]);
    }}
    if (score > bestScore) {{ bestScore = score; best = k; }}
  }}
  return best;
}}
'''

OUT.write_text(header, encoding="utf-8")
print(f"Exported embedded model to {OUT}")
