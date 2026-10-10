from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
RNG = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "embedded" / "snn_model.h"

N_SAMPLES = 1000
TIMESTEPS = 30
N_INPUTS = 2
N_HIDDEN = 4

W_IN = np.array([
    [0.70, 0.25, 0.55, 0.35],
    [0.20, 0.75, 0.40, 0.65],
], dtype=np.float32)
THRESH = np.array([1.20, 1.20, 1.35, 1.35], dtype=np.float32)

labels = RNG.integers(0, 2, size=N_SAMPLES)
spikes = np.zeros((N_SAMPLES, TIMESTEPS, N_INPUTS), dtype=np.float32)
for i, cls in enumerate(labels):
    rates = np.array([0.22, 0.07]) if cls == 0 else np.array([0.07, 0.22])
    spikes[i] = (RNG.random((TIMESTEPS, N_INPUTS)) < rates).astype(np.float32)


def snn_features(sample):
    membrane = np.zeros(N_HIDDEN, dtype=np.float32)
    counts = np.zeros(N_HIDDEN, dtype=np.float32)
    for t in range(TIMESTEPS):
        membrane += sample[t] @ W_IN
        fired = membrane >= THRESH
        counts += fired.astype(np.float32)
        membrane[fired] -= THRESH[fired]
    return counts

features = np.stack([snn_features(sample) for sample in spikes], axis=0)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=SEED, stratify=labels
)

readout = LogisticRegression(random_state=SEED)
readout.fit(X_train, y_train)
pred = readout.predict(X_test)
print("Test accuracy:", round(float(accuracy_score(y_test, pred)), 4))

w_in_text = ", ".join(f"{v:.8f}f" for v in W_IN.reshape(-1))
th_text = ", ".join(f"{v:.8f}f" for v in THRESH)
readout_w = readout.coef_[0]
readout_b = float(readout.intercept_[0])
readout_text = ", ".join(f"{v:.8f}f" for v in readout_w)

header = f"""#pragma once

static const int SNN_TIMESTEPS = {TIMESTEPS};
static const int SNN_INPUTS = {N_INPUTS};
static const int SNN_HIDDEN = {N_HIDDEN};
static const float SNN_W_IN[{N_INPUTS}][{N_HIDDEN}] = {{{{
  {', '.join(f'{v:.8f}f' for v in W_IN[0])}
}}, {{
  {', '.join(f'{v:.8f}f' for v in W_IN[1])}
}}}};
static const float SNN_THRESHOLD[{N_HIDDEN}] = {{{th_text}}};
static const float SNN_READOUT_W[{N_HIDDEN}] = {{{readout_text}}};
static const float SNN_READOUT_B = {readout_b:.8f}f;
"""
OUT.write_text(header, encoding="utf-8")
print(f"Exported SNN parameters to {OUT}")
