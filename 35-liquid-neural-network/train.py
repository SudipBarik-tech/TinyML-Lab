from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
RNG = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "embedded" / "liquid_model.h"

N_SAMPLES = 1000
TIMESTEPS = 24
HIDDEN = 8

labels = RNG.integers(0, 2, size=N_SAMPLES)
t = np.linspace(0.0, 1.0, TIMESTEPS, dtype=np.float32)
X = np.zeros((N_SAMPLES, TIMESTEPS), dtype=np.float32)
for i, cls in enumerate(labels):
    phase = RNG.uniform(-0.4, 0.4)
    carrier = 0.45 * np.sin(2.0 * np.pi * t + phase)
    trend = (0.70 * t - 0.35) if cls == 1 else (-0.70 * t + 0.35)
    noise = RNG.normal(0.0, 0.06, size=TIMESTEPS)
    X[i] = (carrier + trend + noise).astype(np.float32)

W_IN = RNG.normal(0.0, 0.65, size=(HIDDEN,)).astype(np.float32)
W_REC = RNG.normal(0.0, 0.30, size=(HIDDEN, HIDDEN)).astype(np.float32)

# Keep the recurrent dynamics stable by limiting the spectral radius.
eig = np.linalg.eigvals(W_REC.astype(np.float64))
radius = float(np.max(np.abs(eig)))
if radius > 0:
    W_REC *= np.float32(0.85 / radius)

BIAS = RNG.normal(0.0, 0.10, size=(HIDDEN,)).astype(np.float32)
ALPHA = np.linspace(0.10, 0.45, HIDDEN, dtype=np.float32)


def final_state(sequence):
    h = np.zeros(HIDDEN, dtype=np.float32)
    for value in sequence:
        candidate = np.tanh(W_IN * value + W_REC @ h + BIAS).astype(np.float32)
        h = (1.0 - ALPHA) * h + ALPHA * candidate
    return h

states = np.stack([final_state(seq) for seq in X], axis=0)
X_train, X_test, y_train, y_test = train_test_split(
    states, labels, test_size=0.25, random_state=SEED, stratify=labels
)

readout = LogisticRegression(random_state=SEED)
readout.fit(X_train, y_train)
pred = readout.predict(X_test)
print("Test accuracy:", round(float(accuracy_score(y_test, pred)), 4))


def arr(values):
    return ", ".join(f"{float(v):.9f}f" for v in values)

rows = ",\n  ".join("{" + arr(row) + "}" for row in W_REC)
header = f"""#pragma once

static const int LIQUID_TIMESTEPS = {TIMESTEPS};
static const int LIQUID_HIDDEN = {HIDDEN};
static const float LIQUID_W_IN[{HIDDEN}] = {{{arr(W_IN)}}};
static const float LIQUID_W_REC[{HIDDEN}][{HIDDEN}] = {{
  {rows}
}};
static const float LIQUID_BIAS[{HIDDEN}] = {{{arr(BIAS)}}};
static const float LIQUID_ALPHA[{HIDDEN}] = {{{arr(ALPHA)}}};
static const float LIQUID_READOUT_W[{HIDDEN}] = {{{arr(readout.coef_[0])}}};
static const float LIQUID_READOUT_B = {float(readout.intercept_[0]):.9f}f;
"""
OUT.write_text(header, encoding="utf-8")
print(f"Exported liquid-inspired model to {OUT}")
