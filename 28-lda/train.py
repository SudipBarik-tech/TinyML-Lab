from pathlib import Path
import numpy as np
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
OUT = Path(__file__).resolve().parent / "embedded" / "lda_model.h"

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

model = LinearDiscriminantAnalysis(solver="lsqr")
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(float(accuracy_score(y_test, pred)), 4))

coef = np.asarray(model.coef_, dtype=float)
intercept = np.asarray(model.intercept_, dtype=float)
classes = np.asarray(model.classes_, dtype=int)


def c_float_array(name, arr):
    flat = np.asarray(arr).reshape(-1)
    vals = ", ".join(f"{float(v):.9f}f" for v in flat)
    return f"const float {name}[{flat.size}] = {{{vals}}};\n"


def c_int_array(name, arr):
    flat = np.asarray(arr).reshape(-1)
    vals = ", ".join(str(int(v)) for v in flat)
    return f"const int {name}[{flat.size}] = {{{vals}}};\n"

header = "#pragma once\n\n"
header += f"const int LDA_CLASSES = {coef.shape[0]};\n"
header += f"const int LDA_FEATURES = {coef.shape[1]};\n"
header += c_float_array("LDA_COEF", coef)
header += c_float_array("LDA_INTERCEPT", intercept)
header += c_int_array("LDA_LABELS", classes)
header += r'''
inline int ldaPredict(const float *x) {
  int best = 0;
  float bestScore = -1.0e30f;
  for (int c = 0; c < LDA_CLASSES; ++c) {
    float score = LDA_INTERCEPT[c];
    for (int j = 0; j < LDA_FEATURES; ++j) {
      score += LDA_COEF[c * LDA_FEATURES + j] * x[j];
    }
    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }
  return LDA_LABELS[best];
}
'''
OUT.write_text(header, encoding="utf-8")
print(f"Exported embedded model to {OUT}")
