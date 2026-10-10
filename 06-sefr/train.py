from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def fit_sefr(X: np.ndarray, y: np.ndarray):
    pos_mean = X[y == 1].mean(axis=0)
    neg_mean = X[y == 0].mean(axis=0)

    denom = np.abs(pos_mean) + np.abs(neg_mean) + 1e-9
    weights = (pos_mean - neg_mean) / denom

    scores = X @ weights
    candidates = np.unique(scores)
    best_threshold = float(np.median(scores))
    best_accuracy = -1.0

    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        acc = (pred == y).mean()
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)

    return weights, best_threshold


def predict(X: np.ndarray, weights: np.ndarray, threshold: float):
    return (X @ weights >= threshold).astype(int)


def export_header(weights: np.ndarray, threshold: float, output: Path):
    values = ", ".join(f"{v:.8f}f" for v in weights)
    text = f"""#pragma once

const int SEFR_FEATURE_COUNT = {len(weights)};
const float SEFR_WEIGHTS[SEFR_FEATURE_COUNT] = {{{values}}};
const float SEFR_THRESHOLD = {threshold:.8f}f;

inline int sefr_predict(const float *x) {{
  float score = 0.0f;
  for (int i = 0; i < SEFR_FEATURE_COUNT; ++i) {{
    score += x[i] * SEFR_WEIGHTS[i];
  }}
  return score >= SEFR_THRESHOLD ? 1 : 0;
}}
"""
    output.write_text(text, encoding="utf-8")


def main():
    X, y = make_classification(
        n_samples=800,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        class_sep=1.5,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    weights, threshold = fit_sefr(X_train, y_train)
    y_pred = predict(X_test, weights, threshold)

    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Threshold: {threshold:.6f}")

    out = Path(__file__).parent / "embedded" / "sefr_model.h"
    out.parent.mkdir(exist_ok=True)
    export_header(weights, threshold, out)
    print(f"Exported model: {out}")


if __name__ == "__main__":
    main()
