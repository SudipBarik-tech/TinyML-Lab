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
OUT = Path(__file__).resolve().parent / "embedded" / "pruned_model_data.h"

X, y = make_classification(
    n_samples=1400,
    n_features=10,
    n_informative=7,
    n_redundant=3,
    random_state=SEED,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

base = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(24, activation="relu"),
    tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
base.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
base.fit(X_train, y_train, epochs=18, batch_size=32, verbose=0)
_, base_acc = base.evaluate(X_test, y_test, verbose=0)
print("Baseline accuracy:", round(float(base_acc), 4))

end_step = int(np.ceil(len(X_train) / 32.0)) * 8
schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.20,
    final_sparsity=0.70,
    begin_step=0,
    end_step=end_step,
)
pruned = tfmot.sparsity.keras.prune_low_magnitude(base, pruning_schedule=schedule)
pruned.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
pruned.fit(X_train, y_train, epochs=8, batch_size=32, callbacks=callbacks, verbose=0)
_, pruned_acc = pruned.evaluate(X_test, y_test, verbose=0)
print("Pruned accuracy:", round(float(pruned_acc), 4))

stripped = tfmot.sparsity.keras.strip_pruning(pruned)
weights = [w.numpy() for layer in stripped.layers for w in layer.weights if "kernel" in w.name]
all_values = np.concatenate([w.ravel() for w in weights])
sparsity = float(np.mean(all_values == 0.0))
print("Final kernel sparsity:", round(sparsity, 4))

converter = tf.lite.TFLiteConverter.from_keras_model(stripped)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
print("TFLite model size (bytes):", len(tflite_model))

hex_bytes = ", ".join(f"0x{b:02x}" for b in tflite_model)
OUT.write_text(
    f"#pragma once\n\nconst unsigned char PRUNED_MODEL[] = {{{hex_bytes}}};\n"
    f"const unsigned int PRUNED_MODEL_LEN = {len(tflite_model)};\n",
    encoding="utf-8",
)
print(f"Exported model to {OUT}")
