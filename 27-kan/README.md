# Experiment 27 — Compact KAN-Inspired Classifier

## Objective

Study the main idea behind Kolmogorov-Arnold Networks using a compact edge-function model that can still be exported to a microcontroller.

## Important note

This is an educational **KAN-inspired** implementation, not a reproduction of a specific external KAN library. The experiment represents each input feature through a bank of learnable-style univariate basis functions and combines those basis responses with a small linear output layer.

## Method

1. Generate a reproducible nonlinear binary-classification dataset.
2. Standardise the two input features.
3. Apply fixed Gaussian basis functions independently to each feature.
4. Fit the output weights with logistic regression.
5. Export scaling values, basis centres, width and output weights to C/C++.
6. Run the same basis expansion and classification on the device.

## Files

- `train.py` — training, evaluation and compact parameter export.
- `embedded/kan_model.h` — generated parameter header.
- `embedded/kan_demo.ino` — Arduino/ESP32 inference example.

## Run

```bash
python 27-kan/train.py
```

## TinyML observation

A full KAN can become expensive when many edges and spline parameters are used. This experiment keeps the univariate-function idea but limits the number of basis functions so that the model remains easy to inspect and practical for embedded inference.

## Learning outcome

The experiment shows how nonlinear behaviour can be built from small one-dimensional functions attached to individual features instead of relying only on conventional dense-layer activations.
