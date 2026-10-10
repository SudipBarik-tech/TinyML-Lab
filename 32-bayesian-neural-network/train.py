from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "embedded" / "bnn_model.h"

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

X, y = make_classification(
    n_samples=900,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    class_sep=1.1,
    flip_y=0.03,
    random_state=SEED,
)
X_train, X_test, y_train, y_test = train_test_split(
    X.astype(np.float32), y.astype(np.float32),
    test_size=0.25, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

inputs = tf.keras.Input(shape=(5,), name="features")
x = tf.keras.layers.Dense(16, activation="relu")(inputs)
x = tf.keras.layers.Dropout(0.20)(x)
x = tf.keras.layers.Dense(8, activation="relu")(x)
x = tf.keras.layers.Dropout(0.20)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs, name="mc_dropout_network")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=35, batch_size=32, verbose=0)

mc_samples = []
for _ in range(40):
    p = model(X_test, training=True).numpy().reshape(-1)
    mc_samples.append(p)
mc_samples = np.stack(mc_samples, axis=0)
mean_prob = mc_samples.mean(axis=0)
std_prob = mc_samples.std(axis=0)
pred = (mean_prob >= 0.5).astype(int)

print("MC-dropout accuracy:", round(float(accuracy_score(y_test, pred)), 4))
print("Mean predictive std:", round(float(std_prob.mean()), 6))


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
in_scale, in_zero = interpreter.get_input_details()[0]["quantization"]
out_scale, out_zero = interpreter.get_output_details()[0]["quantization"]

byte_text = ", ".join(str(b) for b in model_bytes)
mean_text = ", ".join(f"{v:.9f}f" for v in scaler.mean_)
scale_text = ", ".join(f"{v:.9f}f" for v in scaler.scale_)

header = f"""#pragma once
#include <stdint.h>

static const unsigned char BNN_MODEL[] = {{{byte_text}}};
static const unsigned int BNN_MODEL_LEN = {len(model_bytes)};
static const float BNN_MEAN[5] = {{{mean_text}}};
static const float BNN_SCALE[5] = {{{scale_text}}};
static const float BNN_INPUT_SCALE = {float(in_scale):.12f}f;
static const int BNN_INPUT_ZERO = {int(in_zero)};
static const float BNN_OUTPUT_SCALE = {float(out_scale):.12f}f;
static const int BNN_OUTPUT_ZERO = {int(out_zero)};
"""
OUT.write_text(header, encoding="utf-8")
print(f"Exported deterministic INT8 inference model to {OUT}")
print("Model bytes:", len(model_bytes))
