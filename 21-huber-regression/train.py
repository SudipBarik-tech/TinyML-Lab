from pathlib import Path
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path(__file__).resolve().parent / "embedded" / "huber_model.h"
rng = np.random.default_rng(SEED)

X, y = make_regression(n_samples=500, n_features=4, noise=5.0, random_state=SEED)
outlier_ids = rng.choice(len(y), size=30, replace=False)
y[outlier_ids] += rng.normal(0, 180, size=len(outlier_ids))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

huber = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500).fit(X_train_s, y_train)
linear = LinearRegression().fit(X_train_s, y_train)

for name, model in [("Huber", huber), ("Linear", linear)]:
    pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    print(f"{name} MAE: {mae:.4f} | RMSE: {rmse:.4f}")

coef = huber.coef_.astype(float)
mean = scaler.mean_.astype(float)
scale = scaler.scale_.astype(float)
intercept = float(huber.intercept_)

fmt = lambda arr: ", ".join(f"{v:.9f}f" for v in arr)
header = f'''#pragma once

static const int HUBER_FEATURES = {len(coef)};
static const float HUBER_MEAN[HUBER_FEATURES] = {{{fmt(mean)}}};
static const float HUBER_SCALE[HUBER_FEATURES] = {{{fmt(scale)}}};
static const float HUBER_COEF[HUBER_FEATURES] = {{{fmt(coef)}}};
static const float HUBER_INTERCEPT = {intercept:.9f}f;

inline float huberPredict(const float *x) {{
  float y = HUBER_INTERCEPT;
  for (int i = 0; i < HUBER_FEATURES; ++i) {{
    const float z = (x[i] - HUBER_MEAN[i]) / HUBER_SCALE[i];
    y += HUBER_COEF[i] * z;
  }}
  return y;
}}
'''
OUT.write_text(header, encoding="utf-8")
print(f"Exported model to {OUT}")
