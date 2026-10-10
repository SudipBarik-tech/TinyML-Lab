from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

SEED = 42
OUT = Path(__file__).resolve().parent / "embedded" / "multinomial_nb_model.h"
rng = np.random.default_rng(SEED)

# Count patterns for three synthetic event categories.
class_rates = np.array([
    [8.0, 2.0, 1.0, 1.0, 2.0, 1.0],
    [1.0, 7.0, 5.0, 1.0, 1.0, 2.0],
    [1.0, 1.0, 2.0, 7.0, 5.0, 6.0],
], dtype=float)

samples_per_class = 220
X_parts, y_parts = [], []
for class_id, rate in enumerate(class_rates):
    X_parts.append(rng.poisson(rate, size=(samples_per_class, rate.size)))
    y_parts.append(np.full(samples_per_class, class_id, dtype=int))

X = np.vstack(X_parts)
y = np.concatenate(y_parts)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(float(accuracy_score(y_test, pred)), 4))

priors = model.class_log_prior_
feature_log_prob = model.feature_log_prob_


def c_array(name, array):
    flat = np.asarray(array).reshape(-1)
    values = ", ".join(f"{float(v):.9f}f" for v in flat)
    return f"const float {name}[{len(flat)}] = {{{values}}};\n"

header = "#pragma once\n\n"
header += f"const int MNB_CLASSES = {feature_log_prob.shape[0]};\n"
header += f"const int MNB_FEATURES = {feature_log_prob.shape[1]};\n"
header += c_array("MNB_CLASS_LOG_PRIOR", priors)
header += c_array("MNB_FEATURE_LOG_PROB", feature_log_prob)
header += r'''
inline int multinomialNBPredict(const float *x) {
  int bestClass = 0;
  float bestScore = -1.0e30f;
  for (int c = 0; c < MNB_CLASSES; ++c) {
    float score = MNB_CLASS_LOG_PRIOR[c];
    for (int j = 0; j < MNB_FEATURES; ++j) {
      score += x[j] * MNB_FEATURE_LOG_PROB[c * MNB_FEATURES + j];
    }
    if (score > bestScore) {
      bestScore = score;
      bestClass = c;
    }
  }
  return bestClass;
}
'''
OUT.write_text(header, encoding="utf-8")
print(f"Exported embedded model to {OUT}")
