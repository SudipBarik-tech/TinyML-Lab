from pathlib import Path
import numpy as np

SEED = 42
OUT = Path(__file__).resolve().parent / "embedded" / "gcn_model.h"
rng = np.random.default_rng(SEED)

# Small homophilic graph: nodes 0-3 belong to class 0, nodes 4-7 to class 1.
A = np.array([
    [0,1,1,0,0,0,0,0],
    [1,0,1,1,0,0,0,0],
    [1,1,0,1,0,0,0,0],
    [0,1,1,0,1,0,0,0],
    [0,0,0,1,0,1,1,0],
    [0,0,0,0,1,0,1,1],
    [0,0,0,0,1,1,0,1],
    [0,0,0,0,0,1,1,0],
], dtype=np.float64)

X = np.array([
    [1.00, 0.10, 0.20],
    [0.90, 0.15, 0.10],
    [1.10, 0.05, 0.25],
    [0.80, 0.20, 0.15],
    [0.10, 0.90, 1.00],
    [0.20, 1.00, 0.85],
    [0.05, 1.10, 0.95],
    [0.15, 0.85, 1.10],
], dtype=np.float64)
y = np.array([0,0,0,0,1,1,1,1], dtype=int)

# Symmetric normalisation of A + I.
A_tilde = A + np.eye(A.shape[0])
degree = A_tilde.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

n_nodes, n_features = X.shape
hidden = 5
n_classes = 2

W0 = rng.normal(0.0, 0.35, size=(n_features, hidden))
W1 = rng.normal(0.0, 0.35, size=(hidden, n_classes))

Y = np.eye(n_classes)[y]
lr = 0.08

for epoch in range(1800):
    AX = A_norm @ X
    H_pre = AX @ W0
    H = np.maximum(H_pre, 0.0)
    AH = A_norm @ H
    logits = AH @ W1

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    prob = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    d_logits = (prob - Y) / n_nodes
    dW1 = AH.T @ d_logits
    dH = A_norm.T @ (d_logits @ W1.T)
    dH_pre = dH * (H_pre > 0.0)
    dW0 = AX.T @ dH_pre

    W1 -= lr * dW1
    W0 -= lr * dW0

pred = logits.argmax(axis=1)
accuracy = float((pred == y).mean())
print("Node accuracy:", round(accuracy, 4))
print("Predictions:", pred.tolist())


def c_float_array(name, arr):
    flat = np.asarray(arr).reshape(-1)
    values = ", ".join(f"{float(v):.9f}f" for v in flat)
    return f"const float {name}[{flat.size}] = {{{values}}};\n"

header = "#pragma once\n\n"
header += f"const int GCN_NODES = {n_nodes};\n"
header += f"const int GCN_FEATURES = {n_features};\n"
header += f"const int GCN_HIDDEN = {hidden};\n"
header += f"const int GCN_CLASSES = {n_classes};\n"
header += c_float_array("GCN_A", A_norm)
header += c_float_array("GCN_X", X)
header += c_float_array("GCN_W0", W0)
header += c_float_array("GCN_W1", W1)
header += r'''
inline void gcnPredict(int *labels) {
  float ax[GCN_NODES][GCN_FEATURES] = {0};
  float h[GCN_NODES][GCN_HIDDEN] = {0};
  float ah[GCN_NODES][GCN_HIDDEN] = {0};

  for (int i = 0; i < GCN_NODES; ++i) {
    for (int j = 0; j < GCN_FEATURES; ++j) {
      for (int k = 0; k < GCN_NODES; ++k) {
        ax[i][j] += GCN_A[i * GCN_NODES + k] * GCN_X[k * GCN_FEATURES + j];
      }
    }
  }

  for (int i = 0; i < GCN_NODES; ++i) {
    for (int hidx = 0; hidx < GCN_HIDDEN; ++hidx) {
      float v = 0.0f;
      for (int j = 0; j < GCN_FEATURES; ++j) {
        v += ax[i][j] * GCN_W0[j * GCN_HIDDEN + hidx];
      }
      h[i][hidx] = v > 0.0f ? v : 0.0f;
    }
  }

  for (int i = 0; i < GCN_NODES; ++i) {
    for (int hidx = 0; hidx < GCN_HIDDEN; ++hidx) {
      for (int k = 0; k < GCN_NODES; ++k) {
        ah[i][hidx] += GCN_A[i * GCN_NODES + k] * h[k][hidx];
      }
    }
  }

  for (int i = 0; i < GCN_NODES; ++i) {
    int best = 0;
    float bestScore = -1.0e30f;
    for (int c = 0; c < GCN_CLASSES; ++c) {
      float score = 0.0f;
      for (int hidx = 0; hidx < GCN_HIDDEN; ++hidx) {
        score += ah[i][hidx] * GCN_W1[hidx * GCN_CLASSES + c];
      }
      if (score > bestScore) {
        bestScore = score;
        best = c;
      }
    }
    labels[i] = best;
  }
}
'''
OUT.write_text(header, encoding="utf-8")
print(f"Exported embedded model to {OUT}")
