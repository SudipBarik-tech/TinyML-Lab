from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

SEED = 42
OUT = Path(__file__).resolve().parent / "embedded" / "knn_model.h"

X, y = make_classification(
    n_samples=360, n_features=4, n_informative=4, n_redundant=0,
    n_classes=3, n_clusters_per_class=1, class_sep=1.5, random_state=SEED
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED
)

scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(Xs_train, y_train)
pred = model.predict(Xs_test)
print("Accuracy:", round(float(accuracy_score(y_test, pred)), 4))

# Keep a balanced prototype subset for a smaller embedded footprint.
rng = np.random.default_rng(SEED)
idx = []
for cls in np.unique(y_train):
    cls_idx = np.where(y_train == cls)[0]
    idx.extend(rng.choice(cls_idx, size=min(20, len(cls_idx)), replace=False).tolist())
idx = np.array(idx)
PX = Xs_train[idx]
Py = y_train[idx]

lines = ["#pragma once", "", f"constexpr int KNN_N = {len(PX)};", "constexpr int KNN_F = 4;", "constexpr int KNN_K = 3;"]
lines.append("const float KNN_MEAN[4] = {" + ", ".join(f"{v:.9f}f" for v in scaler.mean_) + "};")
lines.append("const float KNN_SCALE[4] = {" + ", ".join(f"{v:.9f}f" for v in scaler.scale_) + "};")
lines.append("const float KNN_X[KNN_N][KNN_F] = {")
for row in PX:
    lines.append("  {" + ", ".join(f"{v:.9f}f" for v in row) + "},")
lines.append("};")
lines.append("const int KNN_Y[KNN_N] = {" + ", ".join(map(str, Py.tolist())) + "};")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Exported model to {OUT}")
