# 08 — Logistic Regression

## Objective

Build a compact binary classifier whose fitted coefficients can be exported directly to embedded C/C++.

## Method

A four-feature synthetic classification dataset is standardised and used to train logistic regression. The exported model stores scaler statistics, learned coefficients and intercept. Device-side inference reproduces the linear score and sigmoid probability.

## Run

```bash
python 08-logistic-regression/train.py
```

The script reports accuracy and writes `embedded/logistic_model.h`.

## Embedded inference

The microcontroller performs feature standardisation, one weighted sum and a sigmoid operation. This is a useful baseline because its memory requirement grows linearly with the number of input features.

## Files

- `train.py` — reproducible training, evaluation and header export
- `embedded/logistic_demo.ino` — Arduino-compatible inference example

## Observation

Logistic regression is easy to interpret and deploy, but its linear decision boundary can limit performance on strongly nonlinear datasets.
