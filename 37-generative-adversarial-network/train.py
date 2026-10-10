from pathlib import Path
import numpy as np
import tensorflow as tf

SEED = 42
RNG = np.random.default_rng(SEED)
tf.keras.utils.set_random_seed(SEED)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "embedded" / "gan_generator_model.h"

LATENT = 3
HIDDEN = 8
BATCH = 64
STEPS = 1200

angles = RNG.uniform(0.0, 2.0 * np.pi, size=4000)
radius = RNG.normal(1.8, 0.12, size=4000)
real_data = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1).astype(np.float32)

generator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(LATENT,)),
    tf.keras.layers.Dense(HIDDEN, activation="relu"),
    tf.keras.layers.Dense(2),
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

g_opt = tf.keras.optimizers.Adam(2e-3)
d_opt = tf.keras.optimizers.Adam(2e-3)
bce = tf.keras.losses.BinaryCrossentropy()

for step in range(STEPS):
    ids = RNG.integers(0, len(real_data), size=BATCH)
    real = tf.convert_to_tensor(real_data[ids])
    z = tf.random.normal((BATCH, LATENT), seed=SEED + step)

    with tf.GradientTape() as tape_d:
        fake = generator(z, training=True)
        real_score = discriminator(real, training=True)
        fake_score = discriminator(fake, training=True)
        d_loss = bce(tf.ones_like(real_score), real_score) + bce(tf.zeros_like(fake_score), fake_score)
    grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(grads_d, discriminator.trainable_variables))

    z = tf.random.normal((BATCH, LATENT), seed=SEED + 10000 + step)
    with tf.GradientTape() as tape_g:
        fake = generator(z, training=True)
        fake_score = discriminator(fake, training=True)
        g_loss = bce(tf.ones_like(fake_score), fake_score)
    grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(grads_g, generator.trainable_variables))

    if (step + 1) % 300 == 0:
        print(f"step={step + 1} d_loss={float(d_loss):.4f} g_loss={float(g_loss):.4f}")

w1, b1 = generator.layers[0].get_weights()
w2, b2 = generator.layers[1].get_weights()


def arr(name, values):
    flat = values.reshape(-1)
    text = ", ".join(f"{float(v):.9f}f" for v in flat)
    return f"static const float {name}[{len(flat)}] = {{{text}}};\n"

header = "#pragma once\n\n"
header += f"static const int GAN_LATENT = {LATENT};\n"
header += f"static const int GAN_HIDDEN = {HIDDEN};\n"
header += arr("GAN_W1", w1)
header += arr("GAN_B1", b1)
header += arr("GAN_W2", w2)
header += arr("GAN_B2", b2)
header += """
inline void ganGenerate(const float *z, float *out) {
  float h[GAN_HIDDEN];
  for (int j = 0; j < GAN_HIDDEN; ++j) {
    float s = GAN_B1[j];
    for (int i = 0; i < GAN_LATENT; ++i) s += z[i] * GAN_W1[i * GAN_HIDDEN + j];
    h[j] = s > 0.0f ? s : 0.0f;
  }
  for (int j = 0; j < 2; ++j) {
    float s = GAN_B2[j];
    for (int i = 0; i < GAN_HIDDEN; ++i) s += h[i] * GAN_W2[i * 2 + j];
    out[j] = s;
  }
}
"""
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(header, encoding="utf-8")
print(f"Exported generator to {OUT}")
