from pathlib import Path
import json
import re
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

SEED = 42
OUT = Path(__file__).resolve().parent / "embedded" / "xgb_regressor_model.h"

X, y = make_regression(
    n_samples=700,
    n_features=4,
    n_informative=4,
    noise=8.0,
    random_state=SEED,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED
)

model = XGBRegressor(
    n_estimators=12,
    max_depth=3,
    learning_rate=0.15,
    subsample=1.0,
    colsample_bytree=1.0,
    objective="reg:squarederror",
    random_state=SEED,
    n_jobs=1,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
print("MAE:", round(float(mae), 4))
print("RMSE:", round(float(rmse), 4))

booster = model.get_booster()
trees = booster.get_dump(dump_format="json")
config = json.loads(booster.save_config())
base_score_text = str(config["learner"]["learner_model_param"]["base_score"])
base_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", base_score_text)
if base_match is None:
    raise ValueError(f"Could not parse XGBoost base score: {base_score_text!r}")
base_score = float(base_match.group(0))


def emit_node(node, indent="  "):
    if "leaf" in node:
        return f"{indent}return {float(node['leaf']):.9f}f;\n"

    split = str(node["split"])
    m = re.search(r"(\d+)$", split)
    feature_index = int(m.group(1)) if m else int(split)
    threshold = float(node["split_condition"])

    children = {child["nodeid"]: child for child in node["children"]}
    yes_node = children[node["yes"]]
    no_node = children[node["no"]]

    code = f"{indent}if (x[{feature_index}] < {threshold:.9f}f) {{\n"
    code += emit_node(yes_node, indent + "  ")
    code += f"{indent}}} else {{\n"
    code += emit_node(no_node, indent + "  ")
    code += f"{indent}}}\n"
    return code

functions = []
for i, tree_json in enumerate(trees):
    tree = json.loads(tree_json)
    functions.append(
        f"inline float xgbTree{i}(const float *x) {{\n{emit_node(tree, '  ')}}}\n"
    )

aggregate = [f"  float y = {base_score:.9f}f;"]
for i in range(len(trees)):
    aggregate.append(f"  y += xgbTree{i}(x);")
aggregate.append("  return y;")

header = "#pragma once\n\n" + "\n".join(functions)
header += "\ninline float xgbRegress(const float *x) {\n" + "\n".join(aggregate) + "\n}\n"
OUT.write_text(header, encoding="utf-8")
print("Base score:", base_score)
print(f"Exported embedded model to {OUT}")
