from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
RNG = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "embedded" / "rnn_model.h"

TIMESTEPS = 16
FEATURES = 2
N_SAMPLES = 1000

tf.keras.utils.set_random_seed(SEED)

X = np.zeros((N_SAMPLES, TIMESTEPS, FEATURES), dtype=np.float32)
y = RNG.integers(0, 2, size=N_SAMPLES).astype(np.float32)

t = np.linspace(0.0, 2.0 * np.pi, TIMESTEPS, dtype=np.float32)
for i, cls in enumerate(y.astype(int)):
    phase = RNG.uniform(-0.25, 0.25)
    noise = RNG.normal(0.0, 0.08, size=(TIMESTEPS, FEATURES)).astype(np.float32)
    X[i, :, 0] = np.sin(t + phase)
    if cls == 0:
        X[i, :, 1] = np.sin(t + phase + 0.25)
    else:
        X[i, :, 1] = np.sin(t + phase + np.pi)
    X[i] += noise

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(TIMESTEPS, FEATURES)),
    tf.keras.layers.SimpleRNN(12, activation="tanh"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

prob = model.predict(X_test, verbose=0).reshape(-1)
pred = (prob >= 0.5).astype(int)
print("Test accuracy:", round(float(accuracy_score(y_test, pred)), 4))


def representative_data():
    for sample in X_train[:200]:
        yield [sample.reshape(1, TIMESTEPS, FEATURES).astype(np.float32)]

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
header = f"""#pragma once
#include <stdint.h>

static const unsigned char RNN_MODEL[] = {{{byte_text}}};
static const unsigned int RNN_MODEL_LEN = {len(model_bytes)};
static const int RNN_TIMESTEPS = {TIMESTEPS};
static const int RNN_FEATURES = {FEATURES};
static const float RNN_INPUT_SCALE = {float(in_scale):.12f}f;
static const int RNN_INPUT_ZERO = {int(in_zero)};
static const float RNN_OUTPUT_SCALE = {float(out_scale):.12f}f;
static const int RNN_OUTPUT_ZERO = {int(out_zero)};
"""
OUT.write_text(header, encoding="utf-8")
print(f"Exported INT8 model to {OUT}")
print("Model bytes:", len(model_bytes))
