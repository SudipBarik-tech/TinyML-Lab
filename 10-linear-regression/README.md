# 10 — Linear Regression

## Objective

Implement a lightweight regression model for numerical prediction and export the learned coefficients for embedded inference.

## Method

A reproducible four-feature regression dataset is generated in Python. A standard linear-regression model is trained and evaluated using mean absolute error and coefficient of determination (R²). The fitted coefficients and intercept are then exported to a C header.

## Run

```bash
python 10-linear-regression/train.py
```

The script reports test metrics and writes `embedded/linear_regression_model.h`.

## Embedded inference

The device evaluates a simple affine function:

```text
y = b + w1*x1 + w2*x2 + ... + wn*xn
```

No dynamic memory, external ML runtime or iterative optimisation is needed during inference.

## Files

- `train.py` — training, evaluation and header export
- `embedded/linear_regression_demo.ino` — Arduino-compatible prediction example

## Observation

Linear regression is one of the smallest practical regression models for microcontrollers. It works well when the target is approximately linear in the input features, but it cannot directly model complex nonlinear relationships.
