from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "embedded" / "nam_model.h"

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

X, y = make_classification(
    n_samples=900,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    class_sep=1.25,
    random_state=SEED,
)
X_train, X_test, y_train, y_test = train_test_split(
    X.astype(np.float32), y.astype(np.float32),
    test_size=0.25, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

inputs = tf.keras.Input(shape=(4,), name="features")
contributions = []
for i in range(4):
    xi = tf.keras.layers.Lambda(
        lambda z, idx=i: z[:, idx:idx + 1],
        name=f"feature_{i}_slice",
    )(inputs)
    h = tf.keras.layers.Dense(8, activation="relu", name=f"feature_{i}_hidden")(xi)
    ci = tf.keras.layers.Dense(1, name=f"feature_{i}_contribution")(h)
    contributions.append(ci)

logit = tf.keras.layers.Add(name="additive_logit")(contributions)
output = tf.keras.layers.Activation("sigmoid", name="probability")(logit)
model = tf.keras.Model(inputs, output, name="compact_nam")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=35, batch_size=32, verbose=0)

prob = model.predict(X_test, verbose=0).reshape(-1)
pred = (prob >= 0.5).astype(int)
print("Test accuracy:", round(float(accuracy_score(y_test, pred)), 4))


def representative_data():
    for row in X_train[:200]:
        yield [row.reshape(1, -1).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
model_bytes = converter.convert()

interpreter = tf.lite.Interpreter(model_content=model_bytes)
interpreter.allocate_tensors()
in_detail = interpreter.get_input_details()[0]
out_detail = interpreter.get_output_details()[0]
in_scale, in_zero = in_detail["quantization"]
out_scale, out_zero = out_detail["quantization"]

byte_text = ", ".join(str(b) for b in model_bytes)
mean_text = ", ".join(f"{v:.9f}f" for v in scaler.mean_)
scale_text = ", ".join(f"{v:.9f}f" for v in scaler.scale_)
header = f"""#pragma once
#include <stdint.h>

static const unsigned char NAM_MODEL[] = {{{byte_text}}};
static const unsigned int NAM_MODEL_LEN = {len(model_bytes)};
static const float NAM_MEAN[4] = {{{mean_text}}};
static const float NAM_SCALE[4] = {{{scale_text}}};
static const float NAM_INPUT_SCALE = {float(in_scale):.12f}f;
static const int NAM_INPUT_ZERO = {int(in_zero)};
static const float NAM_OUTPUT_SCALE = {float(out_scale):.12f}f;
static const int NAM_OUTPUT_ZERO = {int(out_zero)};
"""
OUT.write_text(header, encoding="utf-8")
print(f"Exported INT8 model to {OUT}")
print("Model bytes:", len(model_bytes))
