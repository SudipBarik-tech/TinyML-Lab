from pathlib import Path
import numpy as np

SEED = 42
RNG = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parent / "embedded" / "som_model.h"

ROWS = 5
COLS = 5
FEATURES = 3
EPOCHS = 40

centers = np.array([
    [-1.5, -1.0, 0.2],
    [1.2, 1.5, -0.6],
    [0.3, -1.3, 1.5],
], dtype=np.float32)

samples = []
for c in centers:
    samples.append(RNG.normal(c, 0.30, size=(250, FEATURES)))
X = np.vstack(samples).astype(np.float32)

w = RNG.normal(0.0, 1.0, size=(ROWS, COLS, FEATURES)).astype(np.float32)
positions = np.stack(np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing="ij"), axis=-1)

for epoch in range(EPOCHS):
    lr = 0.45 * (1.0 - epoch / EPOCHS) + 0.05
    sigma = 2.0 * (1.0 - epoch / EPOCHS) + 0.35
    order = RNG.permutation(len(X))
    for idx in order:
        x = X[idx]
        d = np.sum((w - x) ** 2, axis=2)
        bmu = np.unravel_index(np.argmin(d), d.shape)
        grid_d2 = np.sum((positions - np.array(bmu)) ** 2, axis=2)
        influence = np.exp(-grid_d2 / (2.0 * sigma * sigma)).astype(np.float32)
        w += lr * influence[..., None] * (x - w)

quantisation_error = np.mean([
    np.sqrt(np.min(np.sum((w - x) ** 2, axis=2))) for x in X
])
print("Quantisation error:", round(float(quantisation_error), 4))

flat = w.reshape(-1)
weights = ", ".join(f"{float(v):.9f}f" for v in flat)
header = f"""#pragma once
#include <float.h>

static const int SOM_ROWS = {ROWS};
static const int SOM_COLS = {COLS};
static const int SOM_FEATURES = {FEATURES};
static const float SOM_WEIGHTS[{len(flat)}] = {{{weights}}};

inline int somBestMatchingUnit(const float *x) {{
  float best = FLT_MAX;
  int best_index = 0;
  const int units = SOM_ROWS * SOM_COLS;
  for (int u = 0; u < units; ++u) {{
    float d = 0.0f;
    for (int f = 0; f < SOM_FEATURES; ++f) {{
      float e = x[f] - SOM_WEIGHTS[u * SOM_FEATURES + f];
      d += e * e;
    }}
    if (d < best) {{
      best = d;
      best_index = u;
    }}
  }}
  return best_index;
}}
"""
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(header, encoding="utf-8")
print(f"Exported SOM codebook to {OUT}")
