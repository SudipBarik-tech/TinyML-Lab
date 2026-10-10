from pathlib import Path
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path(__file__).resolve().parent / "embedded" / "elastic_net_model.h"

X, y = make_regression(
    n_samples=500, n_features=6, n_informative=5, noise=10.0, random_state=SEED
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED
)

scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)

model = ElasticNet(alpha=0.08, l1_ratio=0.55, random_state=SEED, max_iter=5000)
model.fit(Xs_train, y_train)
pred = model.predict(Xs_test)
print("MAE:", round(float(mean_absolute_error(y_test, pred)), 4))
print("RMSE:", round(float(mean_squared_error(y_test, pred) ** 0.5), 4))
print("Non-zero coefficients:", int(np.count_nonzero(model.coef_)))

lines = [
    "#pragma once",
    "",
    "constexpr int EN_FEATURES = 6;",
    "const float EN_MEAN[EN_FEATURES] = {" + ", ".join(f"{v:.9f}f" for v in scaler.mean_) + "};",
    "const float EN_SCALE[EN_FEATURES] = {" + ", ".join(f"{v:.9f}f" for v in scaler.scale_) + "};",
    "const float EN_COEF[EN_FEATURES] = {" + ", ".join(f"{v:.9f}f" for v in model.coef_) + "};",
    f"constexpr float EN_INTERCEPT = {float(model.intercept_):.9f}f;",
]
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Exported model to {OUT}")
