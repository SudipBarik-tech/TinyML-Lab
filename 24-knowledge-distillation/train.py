from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
OUT = Path(__file__).resolve().parent / "embedded" / "student_model_data.h"

X, y = make_classification(
    n_samples=1800,
    n_features=12,
    n_informative=9,
    n_redundant=3,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=SEED,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)
y_train_oh = tf.keras.utils.to_categorical(y_train, 3)
y_test_oh = tf.keras.utils.to_categorical(y_test, 3)

teacher = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(12,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(3),
])
teacher.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
teacher.fit(X_train, y_train_oh, epochs=25, batch_size=32, verbose=0)
_, teacher_acc = teacher.evaluate(X_test, y_test_oh, verbose=0)
print("Teacher accuracy:", round(float(teacher_acc), 4))

student = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(12,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3),
])
optimizer = tf.keras.optimizers.Adam(1e-3)
temperature = 3.0
alpha = 0.35
batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_oh)).shuffle(len(X_train), seed=SEED).batch(batch_size)
for epoch in range(20):
    for xb, yb in train_ds:
        teacher_logits = tf.stop_gradient(teacher(xb, training=False))
        with tf.GradientTape() as tape:
            student_logits = student(xb, training=True)
            hard_loss = tf.keras.losses.categorical_crossentropy(yb, student_logits, from_logits=True)
            teacher_soft = tf.nn.softmax(teacher_logits / temperature)
            student_soft = tf.nn.softmax(student_logits / temperature)
            soft_loss = tf.keras.losses.kullback_leibler_divergence(teacher_soft, student_soft) * (temperature ** 2)
            loss = tf.reduce_mean(alpha * hard_loss + (1.0 - alpha) * soft_loss)
        grads = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))

student.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
_, student_acc = student.evaluate(X_test, y_test_oh, verbose=0)
print("Student accuracy:", round(float(student_acc), 4))


def representative_data():
    for i in range(min(200, len(X_train))):
        yield [X_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(student)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
print("Student INT8 model size (bytes):", len(tflite_model))

hex_bytes = ", ".join(f"0x{b:02x}" for b in tflite_model)
OUT.write_text(
    f"#pragma once\n\nconst unsigned char STUDENT_MODEL[] = {{{hex_bytes}}};\n"
    f"const unsigned int STUDENT_MODEL_LEN = {len(tflite_model)};\n",
    encoding="utf-8",
)
print(f"Exported model to {OUT}")
