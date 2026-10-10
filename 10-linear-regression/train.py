from pathlib import Path

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def export_header(model: LinearRegression, output: Path):
    coefs = ", ".join(f"{v:.8f}f" for v in model.coef_)
    intercept = float(model.intercept_)

    text = f"""#pragma once

const int LINREG_FEATURE_COUNT = {len(model.coef_)};
const float LINREG_COEF[LINREG_FEATURE_COUNT] = {{{coefs}}};
const float LINREG_INTERCEPT = {intercept:.8f}f;

inline float linear_regression_predict(const float *x) {{
  float y = LINREG_INTERCEPT;
  for (int i = 0; i < LINREG_FEATURE_COUNT; ++i) {{
    y += x[i] * LINREG_COEF[i];
  }}
  return y;
}}
"""
    output.write_text(text, encoding="utf-8")


def main():
    X, y = make_regression(
        n_samples=800,
        n_features=4,
        n_informative=4,
        noise=8.0,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, pred):.4f}")
    print(f"R2: {r2_score(y_test, pred):.4f}")

    out = Path(__file__).parent / "embedded" / "linear_regression_model.h"
    out.parent.mkdir(exist_ok=True)
    export_header(model, out)
    print(f"Exported model: {out}")


if __name__ == "__main__":
    main()
