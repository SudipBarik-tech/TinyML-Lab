from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

SEED = 42
RNG = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parent / "embedded" / "causal_tree_model.h"

n = 1200
X = RNG.normal(0.0, 1.0, size=(n, 2)).astype(np.float32)
treatment = RNG.integers(0, 2, size=n).astype(np.float32)

baseline = 0.8 * X[:, 0] - 0.4 * X[:, 1]
true_tau = 0.6 + 1.1 * (X[:, 0] > 0.0) - 0.7 * (X[:, 1] > 0.8)
noise = RNG.normal(0.0, 0.35, size=n)
y = baseline + treatment * true_tau + noise

# For randomized treatment with p=0.5, this transformed outcome is an
# unbiased noisy target for the conditional average treatment effect.
p = 0.5
z = y * (treatment - p) / (p * (1.0 - p))

X_train, X_test, z_train, z_test, tau_train, tau_test = train_test_split(
    X, z, true_tau, test_size=0.25, random_state=SEED
)

model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=40, random_state=SEED)
model.fit(X_train, z_train)
pred = model.predict(X_test)
print("MAE against true treatment effect:", round(float(mean_absolute_error(tau_test, pred)), 4))


def emit(node, indent="  "):
    tree = model.tree_
    left = tree.children_left[node]
    right = tree.children_right[node]
    if left == right:
        value = float(tree.value[node].reshape(-1)[0])
        return f"{indent}return {value:.9f}f;\n"
    feature = int(tree.feature[node])
    threshold = float(tree.threshold[node])
    text = f"{indent}if (x[{feature}] <= {threshold:.9f}f) {{\n"
    text += emit(left, indent + "  ")
    text += f"{indent}}} else {{\n"
    text += emit(right, indent + "  ")
    text += f"{indent}}}\n"
    return text

header = "#pragma once\n\ninline float estimateTreatmentEffect(const float *x) {\n"
header += emit(0)
header += "}\n"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(header, encoding="utf-8")
print(f"Exported model to {OUT}")
