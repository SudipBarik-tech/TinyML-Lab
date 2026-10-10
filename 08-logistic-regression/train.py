from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def export_header(model: Pipeline, output: Path):
    scaler = model.named_steps["scale"]
    clf = model.named_steps["clf"]

    means = ", ".join(f"{v:.8f}f" for v in scaler.mean_)
    scales = ", ".join(f"{v:.8f}f" for v in scaler.scale_)
    coefs = ", ".join(f"{v:.8f}f" for v in clf.coef_[0])
    intercept = float(clf.intercept_[0])

    text = f"""#pragma once
#include <math.h>

const int LOGREG_FEATURE_COUNT = {len(scaler.mean_)};
const float LOGREG_MEAN[LOGREG_FEATURE_COUNT] = {{{means}}};
const float LOGREG_SCALE[LOGREG_FEATURE_COUNT] = {{{scales}}};
const float LOGREG_COEF[LOGREG_FEATURE_COUNT] = {{{coefs}}};
const float LOGREG_INTERCEPT = {intercept:.8f}f;

inline float logistic_probability(const float *x) {{
  float score = LOGREG_INTERCEPT;
  for (int i = 0; i < LOGREG_FEATURE_COUNT; ++i) {{
    float z = (x[i] - LOGREG_MEAN[i]) / LOGREG_SCALE[i];
    score += z * LOGREG_COEF[i];
  }}
  return 1.0f / (1.0f + expf(-score));
}}

inline int logistic_predict(const float *x) {{
  return logistic_probability(x) >= 0.5f ? 1 : 0;
}}
"""
    output.write_text(text, encoding="utf-8")


def main():
    X, y = make_classification(
        n_samples=900,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        class_sep=1.4,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, pred):.4f}")

    out = Path(__file__).parent / "embedded" / "logistic_model.h"
    out.parent.mkdir(exist_ok=True)
    export_header(model, out)
    print(f"Exported model: {out}")


if __name__ == "__main__":
    main()
