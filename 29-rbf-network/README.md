# Experiment 29 — Radial Basis Function Network

## Objective

Build a small Radial Basis Function (RBF) classifier and export its centres, width and output weights for embedded inference.

## Method

1. Generate a reproducible nonlinear two-class dataset.
2. Select a compact set of RBF centres using K-Means.
3. Convert each input into Gaussian radial-basis responses.
4. Train a logistic output layer on those responses.
5. Export the fitted centres, width and linear output parameters.
6. Reproduce the same inference path on Arduino/ESP32.

## Files

- `train.py` — centre selection, training, evaluation and export.
- `embedded/rbf_model.h` — generated compact parameter header.
- `embedded/rbf_demo.ino` — device-side inference example.

## Run

```bash
python 29-rbf-network/train.py
```

## TinyML observation

RBF networks are useful for compact nonlinear decision surfaces when the number of centres is kept small. The main embedded cost is the Gaussian distance calculation for each centre. Reducing the number of centres directly reduces memory and inference time.
