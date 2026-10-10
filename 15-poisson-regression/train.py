from pathlib import Path
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path(__file__).resolve().parent / "embedded" / "poisson_model.h"

rng = np.random.default_rng(SEED)
X = rng.normal(size=(700, 3))
true_coef = np.array([0.45, -0.30, 0.25])
rate = np.exp(0.35 + X @ true_coef)
y = rng.poisson(rate).astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

model = PoissonRegressor(alpha=0.05, max_iter=1000)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)

print("MAE:", round(float(mean_absolute_error(y_test, pred)), 4))
print("Mean Poisson deviance:", round(float(mean_poisson_deviance(y_test, pred)), 4))
print("Coefficients:", np.round(model.coef_, 5))
print("Intercept:", round(float(model.intercept_), 5))

fmt = lambda v: f"{float(v):.9f}f"
header = f'''#pragma once
#include <math.h>

static const int POISSON_FEATURES = 3;
static const float POISSON_MEAN[3] = {{{', '.join(fmt(v) for v in scaler.mean_)}}};
static const float POISSON_SCALE[3] = {{{', '.join(fmt(v) for v in scaler.scale_)}}};
static const float POISSON_COEF[3] = {{{', '.join(fmt(v) for v in model.coef_)}}};
static const float POISSON_INTERCEPT = {fmt(model.intercept_)};

inline float poissonPredict(const float *x) {{
  float z = POISSON_INTERCEPT;
  for (int i = 0; i < POISSON_FEATURES; ++i) {{
    const float standardized = (x[i] - POISSON_MEAN[i]) / POISSON_SCALE[i];
    z += standardized * POISSON_COEF[i];
  }}
  return expf(z);
}}
'''

OUT.write_text(header, encoding="utf-8")
print(f"Exported embedded model to {OUT}")
