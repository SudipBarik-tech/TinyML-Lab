# Experiment 12 — Multilayer Perceptron

## Objective
Train a small feed-forward neural network and export its learned weights for microcontroller-side inference without requiring a full neural-network runtime.

## Method
A compact MLP with one hidden layer is trained on a reproducible two-class dataset. The model uses `tanh` activation in the hidden layer and a sigmoid output. The fitted weights, biases and feature-standardisation parameters are exported to `embedded/mlp_model.h`.

## Run
```bash
python 12-mlp/train.py
```

Open `embedded/mlp_demo.ino` after the header has been generated.

## TinyML relevance
A small dense network can be deployed directly as matrix-vector operations. This makes it useful for studying how neural inference works before moving to larger TensorFlow Lite Micro models.

## Output to record
- test accuracy
- hidden-layer size
- exported parameter count
- device inference result
- latency and memory footprint

## Limitation
This example is intentionally small and demonstrates inference mechanics rather than state-of-the-art classification accuracy.
