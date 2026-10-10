from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
OUT = Path(__file__).resolve().parent / "embedded" / "qat_model_data.h"

X, y = make_classification(
    n_samples=1200,
    n_features=8,
    n_informative=6,
    n_redundant=2,
    random_state=SEED,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

base = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
base.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
base.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
_, float_acc = base.evaluate(X_test, y_test, verbose=0)
print("Float accuracy:", round(float(float_acc), 4))

qat = tfmot.quantization.keras.quantize_model(base)
qat.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
qat.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)
_, qat_acc = qat.evaluate(X_test, y_test, verbose=0)
print("QAT accuracy:", round(float(qat_acc), 4))


def representative_data():
    for i in range(min(200, len(X_train))):
        yield [X_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(qat)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
print("INT8 model size (bytes):", len(tflite_model))

hex_bytes = ", ".join(f"0x{b:02x}" for b in tflite_model)
header = f'''#pragma once

const unsigned char QAT_MODEL[] = {{{hex_bytes}}};
const unsigned int QAT_MODEL_LEN = {len(tflite_model)};
'''
OUT.write_text(header, encoding="utf-8")
print(f"Exported model to {OUT}")
